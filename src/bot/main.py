import os
import time
import math
import json
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, List, Deque
from collections import deque

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
# SIMULATION ENV
# =====================================================
INITIAL_USDC = float(os.getenv("INITIAL_USDC", "1000"))
INITIAL_XLM = float(os.getenv("INITIAL_XLM", "0"))

REQUOTE_SECONDS = float(os.getenv("REQUOTE_SECONDS", "3"))
ORDER_TIMEOUT_SECONDS = float(os.getenv("ORDER_TIMEOUT_SECONDS", "30"))

QUEUE_JOIN_FACTOR = float(os.getenv("QUEUE_JOIN_FACTOR", "1.0"))
FILL_RATE_CAP = float(os.getenv("FILL_RATE_CAP", "999999"))

NETWORK_FEE_XLM = float(os.getenv("NETWORK_FEE_XLM", "0.00001"))
OPS_PER_FILL = int(os.getenv("OPS_PER_FILL", "2"))

MAX_INVENTORY_XLM = float(os.getenv("MAX_INVENTORY_XLM", "2000"))
MIN_INVENTORY_XLM = float(os.getenv("MIN_INVENTORY_XLM", "0"))

# =====================================================
# NEW: Rolling stats (48h)
# =====================================================
STATS_FILE = os.getenv("STATS_FILE", "/tmp/stellar_mm_trades.jsonl")
REPORT_EVERY_SECONDS = float(os.getenv("REPORT_EVERY_SECONDS", "60"))
ROLLING_HOURS = float(os.getenv("ROLLING_HOURS", "48"))
ROLLING_WINDOW_SECONDS = ROLLING_HOURS * 3600.0

# We treat each SELL fill as a "realized trade event"
trade_events: Deque[Dict] = deque()  # each: {"ts": float, "profit_usdc": float, ...}

# =====================================================
# Helpers
# =====================================================
def now_str():
    return datetime.now(timezone.utc).strftime("%b %d %Y %H:%M:%S")

def pct(x):
    return f"{x:.4f}%"

def spread_pct(bid, ask):
    return (ask - bid) / bid * 100.0 if bid > 0 else float("nan")

def prune_events(now_ts: float):
    cutoff = now_ts - ROLLING_WINDOW_SECONDS
    while trade_events and trade_events[0]["ts"] < cutoff:
        trade_events.popleft()

def rolling_stats(now_ts: float) -> Tuple[int, float]:
    prune_events(now_ts)
    count = len(trade_events)
    profit = sum(e["profit_usdc"] for e in trade_events)
    return count, profit

def append_event(event: Dict):
    # keep in memory
    trade_events.append(event)
    # best-effort persist
    try:
        with open(STATS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"{now_str()}  WARN could not write STATS_FILE: {e}")

def load_recent_events_on_start():
    # If file exists, load only recent ones
    try:
        if not os.path.exists(STATS_FILE):
            return
        now_ts = time.time()
        cutoff = now_ts - ROLLING_WINDOW_SECONDS
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("ts", 0) >= cutoff:
                        trade_events.append(e)
                except Exception:
                    continue
        prune_events(now_ts)
        if trade_events:
            cnt, prof = rolling_stats(now_ts)
            print(f"{now_str()}  Loaded {cnt} trade events from last {ROLLING_HOURS:.0f}h, profit={prof:.6f} USDC")
    except Exception as e:
        print(f"{now_str()}  WARN could not load STATS_FILE: {e}")

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
    return [(float(lv["price"]), float(lv["amount"])) for lv in levels]

def best_level(levels: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    if not levels:
        return None, None
    return levels[0][0], levels[0][1]

def depth_at_price(levels: List[Tuple[float, float]], price: float) -> float:
    for p, a in levels:
        if p == price:
            return a
    return 0.0

# =====================================================
# Simulation structures
# =====================================================
@dataclass
class SimOrder:
    side: str                # "buy" or "sell"
    price: float             # USDC per XLM
    amount_xlm: float        # remaining
    created_ts: float
    last_depth_at_price: float
    queue_ahead_xlm: float
    filled_xlm: float = 0.0

    def age(self, now_ts: float) -> float:
        return now_ts - self.created_ts

# =====================================================
# Portfolio + Cost basis (avg cost)
# =====================================================
usdc_balance = INITIAL_USDC
xlm_balance = INITIAL_XLM

# average cost per XLM for inventory acquired via buys (USDC per XLM)
inv_xlm = INITIAL_XLM
inv_cost_usdc = 0.0  # total cost basis of current inventory (USDC)

fills = 0
open_buy: Optional[SimOrder] = None
open_sell: Optional[SimOrder] = None
last_requote_ts = 0.0
last_report_ts = 0.0

# =====================================================
# Fill model
# =====================================================
def apply_network_fee():
    global xlm_balance
    xlm_balance -= NETWORK_FEE_XLM * OPS_PER_FILL

def avg_cost_per_xlm() -> float:
    if inv_xlm <= 0:
        return 0.0
    return inv_cost_usdc / inv_xlm

def process_fills(order: SimOrder, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], now_ts: float) -> float:
    """
    Heuristic fill model:
    - depth decrease at our price = consumption
    - consumption first clears queue_ahead_xlm, then fills us
    """
    global usdc_balance, xlm_balance, fills
    global inv_xlm, inv_cost_usdc

    side_levels = bids if order.side == "buy" else asks
    current_depth = depth_at_price(side_levels, order.price)

    delta = order.last_depth_at_price - current_depth  # positive = depth shrank
    if delta > FILL_RATE_CAP:
        delta = FILL_RATE_CAP
    if delta < 0:
        delta = 0.0

    # clear queue ahead first
    consume_q = min(delta, max(order.queue_ahead_xlm, 0.0))
    order.queue_ahead_xlm -= consume_q
    delta -= consume_q

    fill_xlm = min(delta, order.amount_xlm)
    if fill_xlm > 0:
        order.amount_xlm -= fill_xlm
        order.filled_xlm += fill_xlm
        fills += 1
        apply_network_fee()

        if order.side == "buy":
            # spend USDC, acquire XLM into inventory with cost basis
            cost = fill_xlm * order.price
            usdc_balance -= cost
            xlm_balance += fill_xlm

            inv_xlm += fill_xlm
            inv_cost_usdc += cost

        else:
            # sell XLM, realize P&L using avg cost basis
            proceeds = fill_xlm * order.price
            usdc_balance += proceeds
            xlm_balance -= fill_xlm

            # realized profit = proceeds - cost_basis
            cost_per = avg_cost_per_xlm()
            cost_basis = fill_xlm * cost_per
            profit = proceeds - cost_basis

            # remove sold from inventory
            inv_xlm -= fill_xlm
            inv_cost_usdc -= cost_basis
            if inv_xlm < 1e-12:
                inv_xlm = 0.0
                inv_cost_usdc = 0.0

            append_event({
                "ts": now_ts,
                "type": "sell_fill",
                "xlm": fill_xlm,
                "price": order.price,
                "proceeds_usdc": proceeds,
                "cost_basis_usdc": cost_basis,
                "profit_usdc": profit,
            })

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

def can_quote(spread: float, bid_depth: float, ask_depth: float, need_xlm: float) -> Tuple[bool, str]:
    if not math.isfinite(spread) or spread <= 0:
        return False, "bad spread"
    if spread < MIN_SPREAD_PCT:
        return False, "spread too small"
    req = need_xlm * MIN_DEPTH_MULT
    if bid_depth < req or ask_depth < req:
        return False, f"depth too low need={req:.2f} bid={bid_depth:.2f} ask={ask_depth:.2f}"
    return True, "ok"

# =====================================================
# MAIN
# =====================================================
def main():
    global open_buy, open_sell, last_requote_ts, last_report_ts
    global usdc_balance, xlm_balance

    load_recent_events_on_start()

    print("========== STELLAR DEX PAPER BOT (SIMULATED MAKER) ==========")
    print(f"HORIZON_URL={HORIZON_URL}")
    print(f"SELLING: XLM (native)")
    print(f"BUYING : {COUNTER_ASSET_CODE}:{COUNTER_ASSET_ISSUER}")
    print(f"TRADE_COUNTER_AMOUNT={TRADE_COUNTER_AMOUNT} USDC   MIN_SPREAD_PCT={MIN_SPREAD_PCT}")
    print(f"REQUOTE_SECONDS={REQUOTE_SECONDS}  ORDER_TIMEOUT_SECONDS={ORDER_TIMEOUT_SECONDS}")
    print(f"QUEUE_JOIN_FACTOR={QUEUE_JOIN_FACTOR}  MIN_DEPTH_MULT={MIN_DEPTH_MULT}")
    print(f"STATS_FILE={STATS_FILE}  ROLLING_HOURS={ROLLING_HOURS}  REPORT_EVERY_SECONDS={REPORT_EVERY_SECONDS}")
    print(f"START balances: USDC={usdc_balance:.2f} XLM={xlm_balance:.4f}")
    print("=============================================================")

    loops = 0
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

            # quote size in XLM
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

            # ---- periodic 48h report ----
            if (now_ts - last_report_ts) >= REPORT_EVERY_SECONDS:
                last_report_ts = now_ts
                cnt, prof = rolling_stats(now_ts)
                print(f"{now_str()}  REPORT {ROLLING_HOURS:.0f}h trades={cnt} profit={prof:.6f} USDC")

            # ---- inventory safety ----
            if xlm_balance > MAX_INVENTORY_XLM and open_buy:
                open_buy = cancel_order(open_buy, "inventory too high")
            if xlm_balance < MIN_INVENTORY_XLM and open_sell:
                open_sell = cancel_order(open_sell, "inventory too low")

            # ---- quoting logic ----
            ok, why = can_quote(s, bid_depth, ask_depth, quote_xlm)
            should_requote = (now_ts - last_requote_ts) >= REQUOTE_SECONDS

            if ok and should_requote:
                last_requote_ts = now_ts

                # cancel/replace if top moved
                if open_buy and open_buy.price != bid:
                    open_buy = cancel_order(open_buy, "requote to best bid")
                if open_sell and open_sell.price != ask:
                    open_sell = cancel_order(open_sell, "requote to best ask")

                # place BUY at best bid if we have USDC
                if open_buy is None and usdc_balance >= (TRADE_COUNTER_AMOUNT * 1.02):
                    open_buy = place_order("buy", bid, quote_xlm, bids, asks, now_ts)

                # place SELL at best ask if we have XLM inventory to sell
                if open_sell is None and xlm_balance >= quote_xlm:
                    open_sell = place_order("sell", ask, quote_xlm, bids, asks, now_ts)

            elif (not ok) and should_requote:
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

