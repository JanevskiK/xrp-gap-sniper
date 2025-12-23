import os
import time
import math
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, List

# =====================================================
# ENV (your existing variables)
# =====================================================
HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").rstrip("/")

BASE_ASSET_TYPE = os.getenv("BASE_ASSET_TYPE", "native")  # XLM

COUNTER_ASSET_TYPE = os.getenv("COUNTER_ASSET_TYPE", "credit_alphanum4")
COUNTER_ASSET_CODE = os.getenv("COUNTER_ASSET_CODE", "USDC")
COUNTER_ASSET_ISSUER = os.getenv("COUNTER_ASSET_ISSUER", "")

POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
TRADE_COUNTER_AMOUNT = float(os.getenv("TRADE_COUNTER_AMOUNT", "100.0"))  # USDC per quote size

MIN_SPREAD_PCT = float(os.getenv("MIN_SPREAD_PCT", "0.03"))
MIN_DEPTH_MULT = float(os.getenv("MIN_DEPTH_MULT", "1.0"))
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1"))

# =====================================================
# NEW SIMULATION ENV
# =====================================================
INITIAL_USDC = float(os.getenv("INITIAL_USDC", "1000"))
INITIAL_XLM = float(os.getenv("INITIAL_XLM", "0"))

REQUOTE_SECONDS = float(os.getenv("REQUOTE_SECONDS", "3"))
ORDER_TIMEOUT_SECONDS = float(os.getenv("ORDER_TIMEOUT_SECONDS", "30"))

# assume you join behind existing depth at your price (1.0 = full depth ahead)
QUEUE_JOIN_FACTOR = float(os.getenv("QUEUE_JOIN_FACTOR", "1.0"))

# cap on how much depth-change we interpret as "traded" per tick (safety)
FILL_RATE_CAP = float(os.getenv("FILL_RATE_CAP", "999999"))

# network fees: tiny, but modeled
NETWORK_FEE_XLM = float(os.getenv("NETWORK_FEE_XLM", "0.00001"))
OPS_PER_FILL = int(os.getenv("OPS_PER_FILL", "2"))

# inventory risk controls
MAX_INVENTORY_XLM = float(os.getenv("MAX_INVENTORY_XLM", "2000"))
MIN_INVENTORY_XLM = float(os.getenv("MIN_INVENTORY_XLM", "0"))

# =====================================================
# Helpers
# =====================================================
def now_str():
    return datetime.now(timezone.utc).strftime("%b %d %Y %H:%M:%S")

def pct(x):
    return f"{x:.4f}%"

def spread_pct(bid, ask):
    return (ask - bid) / bid * 100.0 if bid > 0 else float("nan")

# =====================================================
# Horizon orderbook
# =====================================================
def get_order_book() -> Dict:
    params = {
        "selling_asset_type": BASE_ASSET_TYPE,      # XLM
        "buying_asset_type": COUNTER_ASSET_TYPE,    # USDC
    }
    if COUNTER_ASSET_TYPE != "native":
        params["buying_asset_code"] = COUNTER_ASSET_CODE
        params["buying_asset_issuer"] = COUNTER_ASSET_ISSUER

    r = requests.get(f"{HORIZON_URL}/order_book", params=params, timeout=10)
    if r.status_code != 200:
        print(f"{now_str()}  HORIZON ERROR {r.status_code}: {r.text}")
        r.raise_for_status()
    return r.json()

def parse_levels(levels: List[Dict]) -> List[Tuple[float, float]]:
    # returns list of (price, amount_base)
    out = []
    for lv in levels:
        out.append((float(lv["price"]), float(lv["amount"])))
    return out

def best_level(levels: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    if not levels:
        return None, None
    return levels[0][0], levels[0][1]

def depth_at_price(levels: List[Tuple[float, float]], price: float) -> float:
    # exact match lookup; Horizon prices are strings -> floats can be exact for these
    for p, a in levels:
        if p == price:
            return a
    return 0.0

# =====================================================
# Order simulation structures
# =====================================================
@dataclass
class SimOrder:
    side: str                # "buy" or "sell"
    price: float             # USDC per XLM
    amount_xlm: float        # desired amount remaining (base)
    created_ts: float
    last_depth_at_price: float
    queue_ahead_xlm: float   # depth ahead of us at same price
    filled_xlm: float = 0.0

    def age(self, now_ts: float) -> float:
        return now_ts - self.created_ts

# =====================================================
# Portfolio
# =====================================================
usdc_balance = INITIAL_USDC
xlm_balance = INITIAL_XLM

realized_pnl_usdc = 0.0
fills = 0

open_buy: Optional[SimOrder] = None
open_sell: Optional[SimOrder] = None

last_requote_ts = 0.0

# =====================================================
# Fill model
# =====================================================
def apply_network_fee():
    global xlm_balance
    # tiny XLM fee
    xlm_balance -= NETWORK_FEE_XLM * OPS_PER_FILL

def process_fills(order: SimOrder, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], now_ts: float) -> float:
    """
    Heuristic:
    - Track depth at our price level over time.
    - If depth decreases, assume that amount was "consumed" by market orders.
    - First it consumes queue ahead; then it can fill us.
    """
    global usdc_balance, xlm_balance, fills, realized_pnl_usdc

    side_levels = bids if order.side == "buy" else asks
    current_depth = depth_at_price(side_levels, order.price)

    delta = order.last_depth_at_price - current_depth  # positive means depth shrank
    # cap to avoid weird spikes
    if delta > FILL_RATE_CAP:
        delta = FILL_RATE_CAP
    if delta < 0:
        delta = 0.0

    # consume queue ahead first
    consume_q = min(delta, max(order.queue_ahead_xlm, 0.0))
    order.queue_ahead_xlm -= consume_q
    delta -= consume_q

    # remaining delta can fill our order
    fill_xlm = min(delta, order.amount_xlm)
    if fill_xlm > 0:
        order.amount_xlm -= fill_xlm
        order.filled_xlm += fill_xlm
        fills += 1
        apply_network_fee()

        if order.side == "buy":
            # buying XLM: spend USDC
            cost = fill_xlm * order.price
            usdc_balance -= cost
            xlm_balance += fill_xlm
        else:
            # selling XLM: receive USDC
            proceeds = fill_xlm * order.price
            usdc_balance += proceeds
            xlm_balance -= fill_xlm

    order.last_depth_at_price = current_depth
    return fill_xlm

def cancel_order(order: Optional[SimOrder], reason: str):
    if order is None:
        return None
    print(f"{now_str()}  CANCEL {order.side.upper()} price={order.price:.7f} remaining_xlm={order.amount_xlm:.4f} reason={reason}")
    return None

def place_order(side: str, price: float, amount_xlm: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], now_ts: float) -> SimOrder:
    levels = bids if side == "buy" else asks
    d = depth_at_price(levels, price)
    q_ahead = d * QUEUE_JOIN_FACTOR
    o = SimOrder(
        side=side,
        price=price,
        amount_xlm=amount_xlm,
        created_ts=now_ts,
        last_depth_at_price=d,
        queue_ahead_xlm=q_ahead,
    )
    print(f"{now_str()}  PLACE {side.upper()} price={price:.7f} amount_xlm={amount_xlm:.4f} queue_ahead={q_ahead:.2f}")
    return o

# =====================================================
# Strategy
# =====================================================
def can_quote(spread: float, bid_depth: float, ask_depth: float, need_xlm: float) -> Tuple[bool, str]:
    if not math.isfinite(spread) or spread <= 0:
        return False, "bad spread"
    if spread < MIN_SPREAD_PCT:
        return False, "spread too small"
    # depth requirement at top
    req = need_xlm * MIN_DEPTH_MULT
    if bid_depth < req or ask_depth < req:
        return False, f"depth too low need={req:.2f} bid={bid_depth:.2f} ask={ask_depth:.2f}"
    return True, "ok"

def main():
    global open_buy, open_sell, last_requote_ts, realized_pnl_usdc

    loops = 0
    print("========== STELLAR DEX PAPER BOT (SIMULATED MAKER) ==========")
    print(f"HORIZON_URL={HORIZON_URL}")
    print(f"SELLING: XLM (native)")
    print(f"BUYING : {COUNTER_ASSET_CODE}:{COUNTER_ASSET_ISSUER}")
    print(f"TRADE_COUNTER_AMOUNT={TRADE_COUNTER_AMOUNT} USDC   MIN_SPREAD_PCT={MIN_SPREAD_PCT}")
    print(f"REQUOTE_SECONDS={REQUOTE_SECONDS}  ORDER_TIMEOUT_SECONDS={ORDER_TIMEOUT_SECONDS}")
    print(f"QUEUE_JOIN_FACTOR={QUEUE_JOIN_FACTOR}  MIN_DEPTH_MULT={MIN_DEPTH_MULT}")
    print(f"START balances: USDC={usdc_balance:.2f} XLM={xlm_balance:.4f}")
    print("=============================================================")

    while True:
        loops += 1
        now_ts = time.time()

        try:
            ob = get_order_book()
            bids = parse_levels(ob.get("bids", []))
            asks = parse_levels(ob.get("asks", []))

            bid, bid_depth = best_level(bids)
            ask, ask_depth = best_level(asks)

            if bid is None or ask is None:
                print(f"{now_str()}  WARN missing bid/ask")
                time.sleep(POLL_SECONDS)
                continue

            s = spread_pct(bid, ask)

            # size: you quote TRADE_COUNTER_AMOUNT worth
            quote_xlm = TRADE_COUNTER_AMOUNT / bid

            if loops % PRINT_EVERY == 0:
                print(
                    f"{now_str()}  TICK bid={bid:.7f} ask={ask:.7f} spread={pct(s)} "
                    f"bid_xlm={bid_depth:.2f} ask_xlm={ask_depth:.2f} "
                    f"balUSDC={usdc_balance:.2f} balXLM={xlm_balance:.4f}"
                )

            # ---- process fills on any open orders ----
            if open_buy:
                filled = process_fills(open_buy, bids, asks, now_ts)
                if filled > 0:
                    print(f"{now_str()}  FILL BUY  +{filled:.4f} XLM @ {open_buy.price:.7f}")

                if open_buy.amount_xlm <= 0:
                    print(f"{now_str()}  DONE BUY fully filled")
                    open_buy = None
                elif open_buy.age(now_ts) > ORDER_TIMEOUT_SECONDS:
                    open_buy = cancel_order(open_buy, "timeout")

            if open_sell:
                filled = process_fills(open_sell, bids, asks, now_ts)
                if filled > 0:
                    print(f"{now_str()}  FILL SELL -{filled:.4f} XLM @ {open_sell.price:.7f}")

                if open_sell.amount_xlm <= 0:
                    print(f"{now_str()}  DONE SELL fully filled")
                    open_sell = None
                elif open_sell.age(now_ts) > ORDER_TIMEOUT_SECONDS:
                    open_sell = cancel_order(open_sell, "timeout")

            # ---- inventory safety ----
            if xlm_balance > MAX_INVENTORY_XLM:
                # too much XLM -> stop quoting buys
                if open_buy:
                    open_buy = cancel_order(open_buy, "inventory too high")
            if xlm_balance < MIN_INVENTORY_XLM:
                # too little XLM -> stop quoting sells
                if open_sell:
                    open_sell = cancel_order(open_sell, "inventory too low")

            # ---- quoting logic: stay at best bid/ask with requotes ----
            ok, why = can_quote(s, bid_depth, ask_depth, quote_xlm)

            should_requote = (now_ts - last_requote_ts) >= REQUOTE_SECONDS

            if ok and should_requote:
                last_requote_ts = now_ts

                # Cancel and replace to stay at top of book
                if open_buy and open_buy.price != bid:
                    open_buy = cancel_order(open_buy, "requote to best bid")
                if open_sell and open_sell.price != ask:
                    open_sell = cancel_order(open_sell, "requote to best ask")

                # Place BUY at best bid if no open buy and we have USDC
                # Cost approx = quote_xlm * bid = TRADE_COUNTER_AMOUNT
                if open_buy is None and usdc_balance >= (TRADE_COUNTER_AMOUNT * 1.02):  # small buffer
                    open_buy = place_order("buy", bid, quote_xlm, bids, asks, now_ts)

                # Place SELL at best ask if no open sell and we have XLM
                if open_sell is None and xlm_balance >= quote_xlm:
                    open_sell = place_order("sell", ask, quote_xlm, bids, asks, now_ts)

            elif (not ok) and should_requote:
                # If conditions no longer good, pull quotes (more realistic)
                last_requote_ts = now_ts
                if open_buy:
                    open_buy = cancel_order(open_buy, f"no quote: {why}")
                if open_sell:
                    open_sell = cancel_order(open_sell, f"no quote: {why}")

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print(f"{now_str()}  ERROR: {e}")
            time.sleep(max(POLL_SECONDS, 1.0))

if __name__ == "__main__":
    main()
