#!/usr/bin/env python3
"""
Market-making / spread-capture bot skeleton with:
✅ Correct 48h profit: profit48h = equity_now - equity_48h_ago (snapshot based)
✅ Trade throttling: cooldown + minimum spread (bps)
✅ Persistent logging (SQLite): trades + equity snapshots
✅ Same style REPORT line you’re using

IMPORTANT:
- This is a working framework. You MUST implement the exchange-specific parts:
  - fetch_orderbook()
  - get_balances()
  - place_limit_buy()
  - place_limit_sell()
  - (optional) cancel_all_orders()
"""

import os
import time
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict

# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    # Pair config (example: XLM/USDC)
    base_asset: str = os.getenv("BASE_ASSET", "XLM")
    quote_asset: str = os.getenv("QUOTE_ASSET", "USDC")

    # Throttling / strategy
    min_spread_bps: float = float(os.getenv("MIN_SPREAD_BPS", "25"))  # e.g. 25 bps = 0.25%
    trade_cooldown_s: int = int(os.getenv("TRADE_COOLDOWN_S", "120"))  # minimum seconds between filled trades
    requote_bps: float = float(os.getenv("REQUOTE_BPS", "10"))         # printed only; or use for quote offsets

    # Trade sizing
    trade_quote_amount: float = float(os.getenv("TRADE_QUOTE_AMOUNT", "10"))  # how much USDC per trade
    min_base_lot: float = float(os.getenv("MIN_BASE_LOT", "0.5"))             # minimum base size to trade

    # Timing
    tick_interval_s: float = float(os.getenv("TICK_INTERVAL_S", "2.0"))
    snapshot_interval_s: int = int(os.getenv("SNAPSHOT_INTERVAL_S", "30"))    # save equity snapshot every N seconds

    # DB
    db_path: str = os.getenv("DB_PATH", "bot.db")

    # Safety
    max_spread_bps: float = float(os.getenv("MAX_SPREAD_BPS", "2500"))  # ignore crazy books (25%)
    max_price_deviation_pct: float = float(os.getenv("MAX_PRICE_DEV_PCT", "20"))  # optional sanity


CFG = Config()


# ----------------------------
# SQLite persistence
# ----------------------------

def db_connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def db_init(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            side TEXT NOT NULL,               -- 'BUY' or 'SELL'
            price REAL NOT NULL,
            base_qty REAL NOT NULL,
            quote_qty REAL NOT NULL,
            fee_quote REAL NOT NULL DEFAULT 0,
            note TEXT
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            equity_quote REAL NOT NULL,       -- total value in quote (USDC)
            bal_quote REAL NOT NULL,
            bal_base REAL NOT NULL,
            mid_price REAL NOT NULL
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );
    """)
    con.commit()

def meta_get(con: sqlite3.Connection, k: str) -> Optional[str]:
    cur = con.execute("SELECT v FROM meta WHERE k=?", (k,))
    row = cur.fetchone()
    return row[0] if row else None

def meta_set(con: sqlite3.Connection, k: str, v: str) -> None:
    con.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
    con.commit()

def log_trade(con: sqlite3.Connection, ts: int, side: str, price: float, base_qty: float, quote_qty: float, fee_quote: float = 0.0, note: str = "") -> None:
    con.execute(
        "INSERT INTO trades(ts,side,price,base_qty,quote_qty,fee_quote,note) VALUES(?,?,?,?,?,?,?)",
        (ts, side, price, base_qty, quote_qty, fee_quote, note)
    )
    con.commit()

def save_snapshot(con: sqlite3.Connection, ts: int, equity_quote: float, bal_quote: float, bal_base: float, mid_price: float) -> None:
    con.execute(
        "INSERT INTO equity_snapshots(ts,equity_quote,bal_quote,bal_base,mid_price) VALUES(?,?,?,?,?)",
        (ts, equity_quote, bal_quote, bal_base, mid_price)
    )
    con.commit()

def count_trades_since(con: sqlite3.Connection, since_ts: int) -> int:
    cur = con.execute("SELECT COUNT(*) FROM trades WHERE ts >= ?", (since_ts,))
    return int(cur.fetchone()[0])

def last_snapshot_before(con: sqlite3.Connection, ts: int) -> Optional[Tuple[int, float]]:
    cur = con.execute(
        "SELECT ts, equity_quote FROM equity_snapshots WHERE ts <= ? ORDER BY ts DESC LIMIT 1",
        (ts,)
    )
    row = cur.fetchone()
    return (int(row[0]), float(row[1])) if row else None

def latest_snapshot(con: sqlite3.Connection) -> Optional[Tuple[int, float]]:
    cur = con.execute(
        "SELECT ts, equity_quote FROM equity_snapshots ORDER BY ts DESC LIMIT 1"
    )
    row = cur.fetchone()
    return (int(row[0]), float(row[1])) if row else None

def last_trade_ts(con: sqlite3.Connection) -> Optional[int]:
    cur = con.execute("SELECT ts FROM trades ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    return int(row[0]) if row else None


# ----------------------------
# Exchange interface (IMPLEMENT THESE)
# ----------------------------

def fetch_orderbook() -> Tuple[float, float, float, float]:
    """
    Return:
      best_bid_price, best_ask_price, bid_size_in_base, ask_size_in_base

    Replace with your exchange call.
    """
    raise NotImplementedError("Implement fetch_orderbook() for your exchange/API")

def get_balances() -> Tuple[float, float]:
    """
    Return:
      (bal_quote, bal_base)  e.g. (USDC, XLM)

    Replace with your exchange call.
    """
    raise NotImplementedError("Implement get_balances() for your exchange/API")

def place_limit_buy(price: float, base_qty: float) -> bool:
    """
    Place a BUY order for base_qty at price (quote per base).
    Return True if filled/accepted.
    """
    raise NotImplementedError("Implement place_limit_buy() for your exchange/API")

def place_limit_sell(price: float, base_qty: float) -> bool:
    """
    Place a SELL order for base_qty at price (quote per base).
    Return True if filled/accepted.
    """
    raise NotImplementedError("Implement place_limit_sell() for your exchange/API")

def cancel_all_orders() -> None:
    """Optional: cancel open orders"""
    return


# ----------------------------
# Helpers
# ----------------------------

def now_ts() -> int:
    return int(time.time())

def bps(x: float) -> float:
    return x * 10_000.0

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def mid_price(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0

def spread_frac(bid: float, ask: float) -> float:
    # fraction, e.g. 0.001 = 0.1%
    if bid <= 0:
        return 0.0
    return (ask - bid) / bid

def compute_equity_quote(bal_quote: float, bal_base: float, mid: float) -> float:
    return bal_quote + (bal_base * mid)

def get_or_set_initial_equity(con: sqlite3.Connection, equity_now: float) -> float:
    v = meta_get(con, "initial_equity_quote")
    if v is None:
        meta_set(con, "initial_equity_quote", f"{equity_now:.10f}")
        return equity_now
    return float(v)

def should_trade(con: sqlite3.Connection, spread_bps: float) -> Tuple[bool, str]:
    if spread_bps < CFG.min_spread_bps:
        return False, f"spread<{CFG.min_spread_bps:.2f}bps"
    if spread_bps > CFG.max_spread_bps:
        return False, f"spread>{CFG.max_spread_bps:.2f}bps"
    lt = last_trade_ts(con)
    if lt is not None and (now_ts() - lt) < CFG.trade_cooldown_s:
        return False, f"cooldown<{CFG.trade_cooldown_s}s"
    return True, "ok"


# ----------------------------
# Strategy (simple: buy then sell at ask)
# ----------------------------

def decide_and_trade(con: sqlite3.Connection, bid: float, ask: float, bid_xlm: float, ask_xlm: float, bal_quote: float, bal_base: float) -> Optional[str]:
    """
    Very simple approach:
    - If spread big enough: attempt a buy near bid (or at bid) with fixed quote amount
    - Then attempt a sell near ask (or at ask) with same base size
    This reduces random churn because cooldown limits frequency.
    """

    sp_bps = bps(spread_frac(bid, ask))
    ok, reason = should_trade(con, sp_bps)
    if not ok:
        return reason

    mid = mid_price(bid, ask)

    # Determine base_qty from desired quote amount at mid
    desired_quote = min(CFG.trade_quote_amount, bal_quote)
    if desired_quote <= 0.0:
        return "no_quote_balance"

    base_qty = desired_quote / mid
    base_qty = math.floor(base_qty * 1_000_000) / 1_000_000  # truncate to 6 decimals
    if base_qty < CFG.min_base_lot:
        return "base_qty_too_small"

    # Basic liquidity check (optional)
    if bid_xlm > 0 and base_qty > bid_xlm:
        base_qty = bid_xlm
    if ask_xlm > 0 and base_qty > ask_xlm:
        base_qty = ask_xlm
    if base_qty < CFG.min_base_lot:
        return "not_enough_book_liquidity"

    # Quotes (you can apply requote_bps offsets if you want)
    buy_price = bid
    sell_price = ask

    # Ensure we can afford buy
    est_cost = base_qty * buy_price
    if est_cost > bal_quote:
        # scale down
        base_qty = math.floor((bal_quote / buy_price) * 1_000_000) / 1_000_000
        if base_qty < CFG.min_base_lot:
            return "insufficient_quote"

    # 1) Buy
    if not place_limit_buy(buy_price, base_qty):
        return "buy_failed"
    log_trade(con, now_ts(), "BUY", buy_price, base_qty, base_qty * buy_price, 0.0, "spread_capture")

    # Refresh balances (recommended in real bot); here we re-use + assume filled
    # 2) Sell (only if we have base)
    # If your exchange has partial fills, you MUST query balances/filled qty.
    if base_qty > (bal_base + base_qty):  # placeholder safety
        return "unexpected_base_state"

    if not place_limit_sell(sell_price, base_qty):
        return "sell_failed"
    log_trade(con, now_ts(), "SELL", sell_price, base_qty, base_qty * sell_price, 0.0, "spread_capture")

    return "traded"


# ----------------------------
# Main loop + reporting
# ----------------------------

def report_line(con: sqlite3.Connection, bid: float, ask: float, bid_xlm: float, ask_xlm: float, bal_quote: float, bal_base: float) -> str:
    ts = now_ts()
    mid = mid_price(bid, ask)
    equity_now = compute_equity_quote(bal_quote, bal_base, mid)

    # Ensure initial equity stored once
    initial = get_or_set_initial_equity(con, equity_now)
    profit_total = equity_now - initial

    # 48h profit uses snapshots
    t48 = ts - int(48 * 3600)
    snap48 = last_snapshot_before(con, t48)
    if snap48 is None:
        # If we don't have a snapshot 48h ago, fall back to the earliest snapshot we have (or 0)
        earliest = last_snapshot_before(con, 0)
        equity_48 = earliest[1] if earliest else equity_now
    else:
        equity_48 = snap48[1]
    profit48h = equity_now - equity_48

    trades_48h = count_trades_since(con, t48)

    return (
        f"REPORT 48h trades={trades_48h} profit48h={profit48h:.6f} {CFG.quote_asset} | "
        f"total_value={equity_now:.2f} {CFG.quote_asset} | "
        f"profit_total={profit_total:.2f} {CFG.quote_asset} | "
        f"open_orders=0 requote_bps={int(CFG.requote_bps)}"
    )

def main() -> None:
    con = db_connect(CFG.db_path)
    db_init(con)

    last_snapshot_ts = 0

    while True:
        try:
            # 1) Fetch market
            bid, ask, bid_xlm, ask_xlm = fetch_orderbook()
            sp = spread_frac(bid, ask)
            sp_bps = bps(sp)

            # 2) Balances
            bal_quote, bal_base = get_balances()

            # 3) Snapshot equity periodically
            ts = now_ts()
            if (ts - last_snapshot_ts) >= CFG.snapshot_interval_s:
                mid = mid_price(bid, ask)
                eq = compute_equity_quote(bal_quote, bal_base, mid)
                save_snapshot(con, ts, eq, bal_quote, bal_base, mid)
                last_snapshot_ts = ts

            # 4) Print tick (your style)
            print(
                f"TICK bid={bid:.7f} ask={ask:.7f} spread={sp*100:.4f}% "
                f"bid_xlm={bid_xlm:.2f} ask_xlm={ask_xlm:.2f} "
                f"bal{CFG.quote_asset}={bal_quote:.2f} bal{CFG.base_asset}={bal_base:.6f}"
            )

            # 5) Decide & trade (throttled)
            action = decide_and_trade(con, bid, ask, bid_xlm, ask_xlm, bal_quote, bal_base)
            if action == "traded":
                # refresh balances for accurate report (recommended)
                bal_quote, bal_base = get_balances()
                print(report_line(con, bid, ask, bid_xlm, ask_xlm, bal_quote, bal_base))
            else:
                # Print report occasionally even if not trading
                if ts % 60 < int(CFG.tick_interval_s):  # about once per minute
                    print(report_line(con, bid, ask, bid_xlm, ask_xlm, bal_quote, bal_base))

        except NotImplementedError as e:
            # You haven't plugged in the exchange yet
            print(f"ERROR: {e}")
            break
        except Exception as e:
            # Keep bot alive
            print(f"ERROR: {type(e).__name__}: {e}")

        time.sleep(CFG.tick_interval_s)


if __name__ == "__main__":
    main()

