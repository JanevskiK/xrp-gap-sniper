import os
import time
import math
import requests
from datetime import datetime, timezone

# =====================================================
# CONFIG (from YOUR ENV variables)
# =====================================================
HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").rstrip("/")

# You set:
# BASE_ASSET_TYPE="native"  (XLM)
# COUNTER_ASSET_TYPE="credit_alphanum4" + USDC + issuer
#
# Horizon order_book uses "selling" and "buying" naming.
# We'll map:
#   SELLING  = BASE   (XLM)
#   BUYING   = COUNTER(USDC)

BASE_ASSET_TYPE = os.getenv("BASE_ASSET_TYPE", "native")

COUNTER_ASSET_TYPE = os.getenv("COUNTER_ASSET_TYPE", "credit_alphanum4")
COUNTER_ASSET_CODE = os.getenv("COUNTER_ASSET_CODE", "USDC")
COUNTER_ASSET_ISSUER = os.getenv("COUNTER_ASSET_ISSUER", "")

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

def best_level(levels):
    """
    Each level is like:
      {"price": "0.2153", "amount": "1234.5678"}
    amount is in SELLING asset units (here: XLM)
    price is in BUYING per SELLING (here: USDC per XLM)
    """
    if not levels:
        return None, None
    return float(levels[0]["price"]), float(levels[0]["amount"])

# =====================================================
# HORIZON ORDER BOOK (FIXED PARAM NAMES)
# =====================================================
def get_order_book():
    # SELLING = base (XLM native)
    # BUYING  = counter (USDC credit)
    params = {
        "selling_asset_type": BASE_ASSET_TYPE,
        "buying_asset_type": COUNTER_ASSET_TYPE,
    }

    # If selling is not native, we'd add selling_asset_code/issuer
    # (not needed for XLM)

    # If buying is not native, we MUST add buying_asset_code/issuer
    if COUNTER_ASSET_TYPE != "native":
        if not COUNTER_ASSET_CODE or not COUNTER_ASSET_ISSUER:
            raise RuntimeError("Missing COUNTER_ASSET_CODE or COUNTER_ASSET_ISSUER env var")
        params["buying_asset_code"] = COUNTER_ASSET_CODE
        params["buying_asset_issuer"] = COUNTER_ASSET_ISSUER

    url = f"{HORIZON_URL}/order_book"
    r = requests.get(url, params=params, timeout=10)

    if r.status_code != 200:
        print(f"{now()}  HORIZON ERROR {r.status_code}: {r.text}")
        r.raise_for_status()

    return r.json()

# =====================================================
# PAPER TRADE (MAKER CAPTURE)
# =====================================================
def paper_trade(bid, bid_depth, ask, ask_depth):
    """
    Your desired model:
    - "Place buy at bid"
    - "Place sell at ask"
    - "Fill if volume exists"

    We simulate one round:
    - Buy XLM at bid using TRADE_COUNTER_AMOUNT (USDC)
    - Sell same XLM at ask
    - Only if top-of-book depth exists on BOTH sides
    """
    global trades, wins, total_net_usdc

    # Convert USDC -> XLM at bid
    xlm_amount = TRADE_COUNTER_AMOUNT / bid
    required_xlm = xlm_amount * MIN_DEPTH_MULT

    # Must have enough depth on both best levels
    if bid_depth < required_xlm or ask_depth < required_xlm:
        print(
            f"{now()}  SKIP depth too low "
            f"(need_xlm={required_xlm:.3f} bid_xlm={bid_depth:.3f} ask_xlm={ask_depth:.3f})"
        )
        return

    spent_usdc = xlm_amount * bid
    received_usdc = xlm_amount * ask
    net = received_usdc - spent_usdc

    trades += 1
    if net > 0:
        wins += 1
    total_net_usdc += net

    print(
        f"{now()}  TRADE XLM={xlm_amount:.4f} "
        f"BUY@bid={bid:.7f} SELL@ask={ask:.7f} "
        f"net={net:.6f} USDC total={total_net_usdc:.6f} "
        f"trades={trades} winrate={(wins/max(trades,1))*100:.1f}%"
    )

# =====================================================
# MAIN LOOP
# =====================================================
def main():
    print("========== STELLAR DEX PAPER BOT (MAKER CAPTURE) ==========")
    print(f"HORIZON_URL={HORIZON_URL}")
    print(f"SELLING: XLM (native)")
    print(f"BUYING : {COUNTER_ASSET_CODE}:{COUNTER_ASSET_ISSUER}")
    print(f"TRADE_COUNTER_AMOUNT={TRADE_COUNTER_AMOUNT} {COUNTER_ASSET_CODE}")
    print(f"MIN_SPREAD_PCT={MIN_SPREAD_PCT}  MIN_DEPTH_MULT={MIN_DEPTH_MULT}")
    print("==========================================================")

    loops = 0
    while True:
        loops += 1
        try:
            ob = get_order_book()

            bid, bid_depth = best_level(ob.get("bids", []))
            ask, ask_depth = best_level(ob.get("asks", []))

            if bid is None or ask is None:
                print(f"{now()}  WARN: missing bid/ask")
                time.sleep(POLL_SECONDS)
                continue

            s = spread_pct(bid, ask)

            if loops % PRINT_EVERY == 0:
                print(
                    f"{now()}  TICK bid={bid:.7f} ask={ask:.7f} "
                    f"spread={pct(s)} bid_xlm={bid_depth:.2f} ask_xlm={ask_depth:.2f}"
                )

            if math.isfinite(s) and s > 0 and s >= MIN_SPREAD_PCT:
                paper_trade(bid, bid_depth, ask, ask_depth)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print(f"{now()}  ERROR: {e}")
            time.sleep(max(POLL_SECONDS, 1.0))

if __name__ == "__main__":
    main()
