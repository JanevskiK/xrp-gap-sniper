import os
import time
import math
import requests
from datetime import datetime, timezone

# =====================================================
# CONFIG (from ENV – exactly matching your variables)
# =====================================================
HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").rstrip("/")

BASE_ASSET_TYPE = os.getenv("BASE_ASSET_TYPE", "native")

COUNTER_ASSET_TYPE = os.getenv("COUNTER_ASSET_TYPE", "credit_alphanum4")
COUNTER_ASSET_CODE = os.getenv("COUNTER_ASSET_CODE", "USDC")
COUNTER_ASSET_ISSUER = os.getenv("COUNTER_ASSET_ISSUER")

POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
TRADE_COUNTER_AMOUNT = float(os.getenv("TRADE_COUNTER_AMOUNT", "100.0"))

MIN_SPREAD_PCT = float(os.getenv("MIN_SPREAD_PCT", "0.03"))
MIN_DEPTH_MULT = float(os.getenv("MIN_DEPTH_MULT", "1.0"))
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1"))

# =====================================================
# STATE
# =====================================================
trades = 0
wins = 0
total_net_usdc = 0.0

# =====================================================
# HELPERS
# =====================================================
def now():
    return datetime.now(timezone.utc).strftime("%b %d %Y %H:%M:%S")

def pct(x):
    return f"{x:.4f}%"

def spread_pct(bid, ask):
    return (ask - bid) / bid * 100.0 if bid > 0 else float("nan")

# =====================================================
# HORIZON ORDER BOOK
# =====================================================
def get_order_book():
    params = {
        "base_asset_type": BASE_ASSET_TYPE,
        "counter_asset_type": COUNTER_ASSET_TYPE,
        "counter_asset_code": COUNTER_ASSET_CODE,
        "counter_asset_issuer": COUNTER_ASSET_ISSUER,
    }

    r = requests.get(f"{HORIZON_URL}/order_book", params=params, timeout=10)

    if r.status_code != 200:
        print(f"{now()}  HORIZON ERROR {r.status_code}: {r.text}")
        r.raise_for_status()

    return r.json()

def best_level(levels):
    if not levels:
        return None, None
    price = float(levels[0]["price"])
    amount_base = float(levels[0]["amount"])  # amount is BASE asset (XLM)
    return price, amount_base

# =====================================================
# PAPER TRADE (MAKER CAPTURE)
# =====================================================
def paper_trade(bid, bid_depth, ask, ask_depth):
    global trades, wins, total_net_usdc

    # Convert USDC → XLM at bid
    base_amount = TRADE_COUNTER_AMOUNT / bid
    required_base = base_amount * MIN_DEPTH_MULT

    if bid_depth < required_base or ask_depth < required_base:
        print(
            f"{now()}  SKIP depth too low "
            f"(need={required_base:.3f} bid={bid_depth:.3f} ask={ask_depth:.3f})"
        )
        return

    spent = base_amount * bid
    received = base_amount * ask
    net = received - spent

    trades += 1
    if net > 0:
        wins += 1
    total_net_usdc += net

    print(
        f"{now()}  TRADE XLM={base_amount:.4f} "
        f"BUY@{bid:.7f} SELL@{ask:.7f} "
        f"net={net:.6f} USDC "
        f"total={total_net_usdc:.6f} "
        f"trades={trades} winrate={(wins/max(trades,1))*100:.1f}%"
    )

# =====================================================
# MAIN LOOP
# =====================================================
def main():
    print("========== STELLAR DEX PAPER BOT ==========")
    print(f"HORIZON_URL={HORIZON_URL}")
    print(f"PAIR: XLM / {COUNTER_ASSET_CODE}")
    print(f"TRADE_COUNTER_AMOUNT={TRADE_COUNTER_AMOUNT} USDC")
    print(f"MIN_SPREAD_PCT={MIN_SPREAD_PCT}  MIN_DEPTH_MULT={MIN_DEPTH_MULT}")
    print("==========================================")

    loops = 0

    while True:
        loops += 1
        try:
            ob = get_order_book()

            bid, bid_depth = best_level(ob.get("bids", []))
            ask, ask_depth = best_level(ob.get("asks", []))

            if bid is None or ask is None:
                print(f"{now()}  WARN no bid/ask")
                time.sleep(POLL_SECONDS)
                continue

            spread = spread_pct(bid, ask)

            if loops % PRINT_EVERY == 0:
                print(
                    f"{now()}  TICK bid={bid:.7f} ask={ask:.7f} "
                    f"spread={pct(spread)} "
                    f"bid_depth={bid_depth:.2f} ask_depth={ask_depth:.2f}"
                )

            if spread >= MIN_SPREAD_PCT:
                paper_trade(bid, bid_depth, ask, ask_depth)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print(f"{now()}  ERROR {e}")
            time.sleep(max(POLL_SECONDS, 1.0))

if __name__ == "__main__":
    main()
