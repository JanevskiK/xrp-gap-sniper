import os
import time
import math
import requests
from datetime import datetime

# --------------------
# ENV / CONFIG
# --------------------
HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").rstrip("/")

BASE_ASSET_TYPE = os.getenv("BASE_ASSET_TYPE", "native")  # XLM

COUNTER_ASSET_TYPE = os.getenv("COUNTER_ASSET_TYPE", "credit_alphanum4")
COUNTER_ASSET_CODE = os.getenv("COUNTER_ASSET_CODE", "USDC")
COUNTER_ASSET_ISSUER = os.getenv(
    "COUNTER_ASSET_ISSUER",
    "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
)

POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
TRADE_COUNTER_AMOUNT = float(os.getenv("TRADE_COUNTER_AMOUNT", "10.0"))  # amount in USDC

MIN_SPREAD_PCT = float(os.getenv("MIN_SPREAD_PCT", "0.03"))  # spread % threshold to consider
MIN_DEPTH_MULT = float(os.getenv("MIN_DEPTH_MULT", "1.0"))   # require depth >= trade_size * mult

PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1"))

# --------------------
# STATE
# --------------------
trades = 0
wins = 0
total_net_counter = 0.0  # net P&L in counter asset (USDC)

# --------------------
# HELPERS
# --------------------
def now_str():
    return datetime.utcnow().strftime("%b %d %Y %H:%M:%S")

def pct(x):
    return f"{x:.4f}%"

def get_order_book():
    """
    Horizon order book endpoint:
    GET /order_book?base_asset_type=...&counter_asset_type=... (etc)
    If base or counter is native (XLM), only type=native is needed. :contentReference[oaicite:2]{index=2}
    """
    params = {
        "base_asset_type": BASE_ASSET_TYPE,
        "counter_asset_type": COUNTER_ASSET_TYPE,
        "counter_asset_code": COUNTER_ASSET_CODE,
        "counter_asset_issuer": COUNTER_ASSET_ISSUER,
    }

    url = f"{HORIZON_URL}/order_book"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def best_level(levels):
    """
    levels are dicts like:
    {'price': '0.2153', 'amount': '1234.5678'}  (amount is in base asset)
    """
    if not levels:
        return None, None
    p = float(levels[0]["price"])
    a = float(levels[0]["amount"])
    return p, a

def spread_pct_from_bid_ask(bid, ask):
    if bid <= 0 or ask <= 0:
        return float("nan")
    return (ask - bid) / bid * 100.0

def paper_capture_spread(bid, bid_amt_base, ask, ask_amt_base):
    """
    PAPER MODE (optimistic):
    - BUY base (XLM) at bid price (maker buy)
    - SELL base (XLM) at ask price (maker sell)
    - 'fill if volume exists' -> we require depth on both best levels

    bid/ask prices are in COUNTER per BASE (USDC per XLM)
    amount in orderbook is BASE amount available at that price
    """
    global trades, wins, total_net_counter

    # Convert desired counter amount to base amount at bid
    desired_base = TRADE_COUNTER_AMOUNT / bid  # XLM

    # Depth checks at best levels
    required_base = desired_base * MIN_DEPTH_MULT
    if bid_amt_base < required_base or ask_amt_base < required_base:
        return False, f"DEPTH too low: need_base={required_base:.6f} bid_base={bid_amt_base:.6f} ask_base={ask_amt_base:.6f}"

    # Simulate buy at bid (spend counter)
    base_bought = desired_base
    counter_spent = base_bought * bid

    # Simulate sell at ask (receive counter)
    counter_received = base_bought * ask

    net = counter_received - counter_spent  # profit in USDC
    trades += 1
    if net > 0:
        wins += 1
    total_net_counter += net

    msg = (
        f"TRADE base={base_bought:.6f} XLM | "
        f"BUY@bid {bid:.7f} -> spent={counter_spent:.6f} {COUNTER_ASSET_CODE} | "
        f"SELL@ask {ask:.7f} -> got={counter_received:.6f} {COUNTER_ASSET_CODE} | "
        f"net={net:.6f} total_net={total_net_counter:.6f} trades={trades} winrate={(wins/max(trades,1))*100:.1f}%"
    )
    return True, msg

def main():
    print("========== STELLAR DEX PAPER BOT (maker capture) ==========")
    print(f"HORIZON_URL={HORIZON_URL}")
    print(f"PAIR: XLM (native) / {COUNTER_ASSET_CODE}:{COUNTER_ASSET_ISSUER}")
    print(f"TRADE_COUNTER_AMOUNT={TRADE_COUNTER_AMOUNT} {COUNTER_ASSET_CODE}")
    print(f"MIN_SPREAD_PCT={MIN_SPREAD_PCT}  MIN_DEPTH_MULT={MIN_DEPTH_MULT}")
    print("===========================================================")

    loops = 0
    while True:
        loops += 1
        try:
            ob = get_order_book()
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            # Horizon orderbook: bids/asks have 'price' and 'amount' (amount is BASE asset)
            bid, bid_amt_base = best_level(bids)  # best bid price + base depth
            ask, ask_amt_base = best_level(asks)  # best ask price + base depth

            if bid is None or ask is None:
                print(f"{now_str()}  WARN: missing bid/ask")
                time.sleep(POLL_SECONDS)
                continue

            spread = spread_pct_from_bid_ask(bid, ask)

            if loops % PRINT_EVERY == 0:
                print(
                    f"{now_str()}  TICK bid={bid:.7f} ask={ask:.7f} "
                    f"spread={pct(spread)} bid_base={bid_amt_base:.3f} ask_base={ask_amt_base:.3f}"
                )

            if not math.isfinite(spread) or spread <= 0:
                time.sleep(POLL_SECONDS)
                continue

            if spread >= MIN_SPREAD_PCT:
                ok, msg = paper_capture_spread(bid, bid_amt_base, ask, ask_amt_base)
                if ok:
                    print(f"{now_str()}  {msg}")
                else:
                    print(f"{now_str()}  SKIP {msg}")

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print(f"{now_str()}  ERROR: {e}")
            time.sleep(max(POLL_SECONDS, 1.0))

if __name__ == "__main__":
    main()
