import os
import time
import math
import traceback
from datetime import datetime

import ccxt

# =========================
# CONFIG (via env vars)
# =========================
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
SYMBOL = os.getenv("SYMBOL", "XLM/USDC")

PAPER = os.getenv("PAPER", "true").lower() in ("1", "true", "yes", "y")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))

TRADE_USDC = float(os.getenv("TRADE_USDC", "10.0"))

# Spread signal threshold you had before (still used as a baseline)
SPREAD_THRESHOLD_PCT = float(os.getenv("SPREAD_THRESHOLD_PCT", "0.02"))

# --- Step 7 knobs ---
# Fees are the big killer. There are 2 ways to set them:
#   A) Set FEE_PER_SIDE_PCT (e.g., 0.10 means 0.10% each side => 0.20% round trip)
#   B) OR set ROUND_TRIP_FEE_PCT directly
FEE_PER_SIDE_PCT = float(os.getenv("FEE_PER_SIDE_PCT", "0.10"))  # 0.10% default
ROUND_TRIP_FEE_PCT = os.getenv("ROUND_TRIP_FEE_PCT", "").strip()
if ROUND_TRIP_FEE_PCT:
    ROUND_TRIP_FEE_PCT = float(ROUND_TRIP_FEE_PCT)
else:
    ROUND_TRIP_FEE_PCT = 2.0 * FEE_PER_SIDE_PCT

# Additional profit cushion required on top of fees (e.g., 0.05% or 0.10%)
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.05"))

# Safety: do not trade if spread is crazy (bad data)
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "5.0"))

# Print ticks every N loops
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "1"))

# API keys only needed if PAPER=false
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# =========================
# State
# =========================
total_net = 0.0
trades = 0
wins = 0

# =========================
# Helpers
# =========================
def now_str():
    return datetime.utcnow().strftime("%b %d %Y %H:%M:%S")

def pct(x):
    return f"{x:.4f}%"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def compute_spread_pct(bid: float, ask: float) -> float:
    # percent spread based on bid (same as your logs)
    if bid <= 0 or ask <= 0:
        return float("nan")
    return (ask - bid) / bid * 100.0

def compute_required_spread_pct() -> float:
    # ✅ Step 7: fee-aware profitability gate
    return ROUND_TRIP_FEE_PCT + MIN_PROFIT_PCT

def round_down(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step

# =========================
# Exchange setup
# =========================
def make_exchange():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "enableRateLimit": True,
    })
    if not PAPER:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("LIVE trading requires API_KEY and API_SECRET env vars.")
        ex.apiKey = API_KEY
        ex.secret = API_SECRET
    return ex

# =========================
# Trading functions
# =========================
def paper_trade(bid: float, ask: float, spread_pct: float):
    """
    Simulate: BUY at ask then SELL at bid immediately.
    This still *crosses* the spread, but now we only allow it when spread >= fees+profit.
    """
    global total_net, trades, wins

    # BUY at ask
    xlm = TRADE_USDC / ask

    # SELL at bid
    usdc_out = xlm * bid

    gross = usdc_out - TRADE_USDC  # will be negative by ~spread cost

    # Fee model: percent of notional each side
    # fee_usdc ~= trade_usdc * fee% (buy) + usdc_out * fee% (sell)
    fee_buy = TRADE_USDC * (FEE_PER_SIDE_PCT / 100.0)
    fee_sell = usdc_out * (FEE_PER_SIDE_PCT / 100.0)
    fee_total = fee_buy + fee_sell

    net = gross - fee_total

    trades += 1
    if net > 0:
        wins += 1
    total_net += net

    print(f"{now_str()}  BUY  USDC={TRADE_USDC:.2f} price(ask)={ask:.6f} -> XLM={xlm:.7f}")
    print(f"{now_str()}  SELL XLM={xlm:.7f} price(bid)={bid:.6f} -> USDC_out={usdc_out:.6f}")
    print(f"{now_str()}  P&L gross={gross:.6f} fee={fee_total:.6f} net={net:.6f} total_net={total_net:.6f} trades={trades} winrate={(wins/max(trades,1))*100:.1f}%")

def live_trade(exchange, bid: float, ask: float):
    """
    LIVE mode (simple market in/out) — still not a great strategy, but included because you asked for whole code.
    WARNING: Market in/out will usually lose unless spread is huge and fees are tiny.
    """
    global total_net, trades, wins

    # Market BUY by quote amount is not supported consistently across exchanges,
    # so we compute base size from TRADE_USDC and ask.
    base_amount = TRADE_USDC / ask

    # Some markets have amount precision; try to respect it.
    market = exchange.market(SYMBOL)
    amount_step = None
    # ccxt doesn't always give "step"; use precision if present
    precision_amt = market.get("precision", {}).get("amount", None)
    if precision_amt is not None:
        # If precision is decimals, round down to that decimals
        base_amount = float(f"{base_amount:.{precision_amt}f}")

    print(f"{now_str()}  LIVE BUY  {SYMBOL} amount={base_amount}")

    buy_order = exchange.create_market_buy_order(SYMBOL, base_amount)

    # Small pause (optional)
    time.sleep(0.5)

    print(f"{now_str()}  LIVE SELL {SYMBOL} amount={base_amount}")
    sell_order = exchange.create_market_sell_order(SYMBOL, base_amount)

    # We can’t perfectly compute realized P&L without fills/fees parsing (varies by exchange).
    # We'll just count the trade as executed.
    trades += 1
    print(f"{now_str()}  LIVE DONE trades={trades} (PnL requires fill/fee parsing per exchange)")

# =========================
# Main loop
# =========================
def main():
    print("========== BOT START ==========")
    print(f"EXCHANGE={EXCHANGE_ID} SYMBOL={SYMBOL} PAPER={PAPER}")
    print(f"TRADE_USDC={TRADE_USDC} POLL_SECONDS={POLL_SECONDS}")
    print(f"SPREAD_THRESHOLD_PCT={SPREAD_THRESHOLD_PCT}")
    print(f"FEE_PER_SIDE_PCT={FEE_PER_SIDE_PCT} ROUND_TRIP_FEE_PCT={ROUND_TRIP_FEE_PCT} MIN_PROFIT_PCT={MIN_PROFIT_PCT}")
    print(f"STEP7 REQUIRED_SPREAD_PCT={compute_required_spread_pct():.4f}%")
    print("===============================")

    ex = make_exchange()
    loops = 0

    while True:
        try:
            loops += 1

            # Pull best bid/ask
            ob = ex.fetch_order_book(SYMBOL, limit=5)
            bid = safe_float(ob["bids"][0][0]) if ob.get("bids") else float("nan")
            ask = safe_float(ob["asks"][0][0]) if ob.get("asks") else float("nan")

            spread_pct = compute_spread_pct(bid, ask)

            if loops % PRINT_EVERY == 0:
                bids_n = len(ob.get("bids", []))
                asks_n = len(ob.get("asks", []))
                print(f"{now_str()}  TICK bid={bid:.6f} ask={ask:.6f} spread={pct(spread_pct)} bids={bids_n} asks={asks_n}")

            # Sanity checks
            if not (math.isfinite(bid) and math.isfinite(ask) and math.isfinite(spread_pct)):
                time.sleep(POLL_SECONDS)
                continue

            if spread_pct <= 0 or spread_pct > MAX_SPREAD_PCT:
                # likely junk / stale / crazy
                time.sleep(POLL_SECONDS)
                continue

            # You can keep your original threshold, but step 7 adds the REAL gate:
            # ✅ Only trade if spread clears BOTH:
            #   - your signal threshold
            #   - required spread to cover fees + profit
            required_spread = compute_required_spread_pct()
            do_signal = spread_pct >= SPREAD_THRESHOLD_PCT
            do_profit_gate = spread_pct >= required_spread

            if do_signal:
                print(f"{now_str()}  SIGNAL spread {pct(spread_pct)} >= threshold {pct(SPREAD_THRESHOLD_PCT)}")

            # ✅ STEP 7: fee-aware check
            print(f"{now_str()}  CHECK  spread={pct(spread_pct)} required={pct(required_spread)} -> {'TRADE' if (do_signal and do_profit_gate) else 'SKIP'}")

            if do_signal and do_profit_gate:
                if PAPER:
                    print(f"{now_str()}  PAPER TRADE")
                    paper_trade(bid, ask, spread_pct)
                else:
                    print(f"{now_str()}  LIVE TRADE (market in/out) — be careful")
                    live_trade(ex, bid, ask)

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print(f"{now_str()}  ERROR: {e}")
            traceback.print_exc()
            time.sleep(max(POLL_SECONDS, 1.0))

if __name__ == "__main__":
    main()
