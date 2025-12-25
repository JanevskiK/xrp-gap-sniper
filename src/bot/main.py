#!/usr/bin/env python3
import os
import time
import math
import sqlite3
import random
from dataclasses import dataclass
from typing import Optional, Tuple

# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    base_asset: str = os.getenv("BASE_ASSET", "XLM")
    quote_asset: str = os.getenv("QUOTE_ASSET", "USDC")

    # Throttle trading a lot (these two kill the 3,000+ trades problem)
    min_spread_bps: float = float(os.getenv("MIN_SPREAD_BPS", "30"))  # 30 bps = 0.30%
    trade_cooldown_s: int = int(os.getenv("TRADE_COOLDOWN_S", "120"))  # 2 minutes

    # trade size in quote (USDC) used per cycle
    trade_quote_amount: float = float(os.getenv("TRADE_QUOTE_AMOUNT", "10"))
    min_base_lot: float = float(os.getenv("MIN_BASE_LOT", "0.5"))

    # Paper market simulation
    start_price: float = float(os.getenv("START_PRICE", "0.215"))
    sim_vol_bps: float = float(os.getenv("SIM_VOL_BPS", "8"))   # random walk step size (bps)
    sim_spread_bps: float = float(os.getenv("SIM_SPREAD_BPS", "20"))  # typical spread in bps

    # Timing
    tick_interval_s: float = float(os.getenv("TICK_INTERVAL_S", "2.0"))
    snapshot_interval_s: int = int(os.getenv("SNAPSHOT_INTERVAL_S", "30"))

    # Reporting
    requote_bps: float = float(os.getenv("REQUOTE_BPS", "10"))

    # Starting balances (paper)
    starting_quote: float = float(os.getenv("STARTING_QUOTE", "1000"))
    starting_base: float = float(os.getenv("STARTING_BASE", "0"))

    # DB
    db_path: str = os.getenv("DB_PATH", "bot.db")

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
            side TEXT NOT NULL,
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
            equity_quote REAL NOT NULL,
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

def last_trade_ts(con: sqlite3.Connection) -> Optional[int]:
    cur = con.execute("SELECT ts FROM trades ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    return int(row[0]) if row else None


# ----------------------------
# Helpers
# ----------------------------

def now_ts() -> int:
    return int(time.time())

def bps_from_frac(x: float) -> float:
    return x * 10_000.0

def mid_price(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0

def spread_frac(bid: float, ask: float) -> float:
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
    lt = last_trade_ts(con)
    if lt is not None and (now_ts() - lt) < CFG.trade_cooldown_s:
        return False, f"cooldown<{CFG.trade_cooldown_s}s"
    return True, "ok"


# ----------------------------
# PAPER market + PAPER balances
# ----------------------------

class PaperMarket:
    def __init__(self, start: float):
        self.mid = start

    def step(self) -> Tuple[float, float, float, float]:
        # random walk on mid
        step_bps = random.uniform(-CFG.sim_vol_bps, CFG.sim_vol_bps)
        self.mid *= (1.0 + step_bps / 10_000.0)

        # create spread around mid
        half_spread = max(1.0, CFG.sim_spread_bps) / 2.0  # in bps
        bid = self.mid * (1.0 - half_spread / 10_000.0)
        ask = self.mid * (1.0 + half_spread / 10_000.0)

        # liquidity (just for printing)
        bid_xlm = random.uniform(50, 5000)
        ask_xlm = random.uniform(50, 5000)
        return bid, ask, bid_xlm, ask_xlm

class PaperWallet:
    def __init__(self, quote: float, base: float):
        self.quote = quote
        self.base = base

PAPER_MARKET = PaperMarket(CFG.start_price)
PAPER_WALLET = PaperWallet(CFG.starting_quote, CFG.starting_base)


def fetch_orderbook() -> Tuple[float, float, float, float]:
    return PAPER_MARKET.step()

def get_balances() -> Tuple[float, float]:
    return PAPER_WALLET.quote, PAPER_WALLET.base

def place_limit_buy(price: float, base_qty: float) -> bool:
    # Paper fill = immediate fill at provided price
    cost = price * base_qty
    if cost > PAPER_WALLET.quote:
        return False
    PAPER_WALLET.quote -= cost
    PAPER_WALLET.base += base_qty
    return True

def place_limit_sell(price: float, base_qty: float) -> bool:
    if base_qty > PAPER_WALLET.base:
        return False
    proceeds = price * base_qty
    PAPER_WALLET.base -= base_qty
    PAPER_WALLET.quote += proceeds
    return True


# ----------------------------
# Strategy: one round-trip when conditions meet
# ----------------------------

def decide_and_trade(con: sqlite3.Connection, bid: float, ask: float, bid_xlm: float, ask_xlm: float, bal_quote: float, bal_base: float) -> str:
    sp_bps = bps_from_frac(spread_frac(bid, ask))
    ok, reason = should_trade(con, sp_bps)
    if not ok:
        return reason

    mid = mid_price(bid, ask)
    desired_quote = min(CFG.trade_quote_amount, bal_quote)
    if desired_quote <= 0:
        return "no_quote_balance"

    base_qty = desired_quote / mid
    base_qty = math.floor(base_qty * 1_000_000) / 1_000_000
    if base_qty < CFG.min_base_lot:
        return "base_qty_too_small"

    # Buy at bid, sell at ask (paper immediate)
    if not place_limit_buy(bid, base_qty):
        return "buy_failed"
    log_trade(con, now_ts(), "BUY", bid, base_qty, base_qty * bid, 0.0, "paper")

    if not place_limit_sell(ask, base_qty):
        return "sell_failed"
    log_trade(con, now_ts(), "SELL", ask, base_qty, base_qty * ask, 0.0, "paper")

    return "traded"


# ----------------------------
# Reporting
# ----------------------------

def report_line(con: sqlite3.Connection, bid: float, ask: float, bid_xlm: float, ask_xlm: float, bal_quote: float, bal_base: float) -> str:
    ts = now_ts()
    mid = mid_price(bid, ask)
    equity_now = compute_equity_quote(bal_quote, bal_base, mid)

    initial = get_or_set_initial_equity(con, equity_now)
    profit_total = equity_now - initial

    t48 = ts - int(48 * 3600)
    snap48 = last_snapshot_before(con, t48)
    if snap48 is None:
        # not enough history yet, use earliest snapshot if exists, otherwise treat as 0
        snap0 = last_snapshot_before(con, 0)
        equity_48 = snap0[1] if snap0 else equity_now
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


# ----------------------------
# Main loop
# ----------------------------

def main() -> None:
    con = db_connect(CFG.db_path)
    db_init(con)

    last_snapshot_ts = 0

    while True:
        try:
            bid, ask, bid_xlm, ask_xlm = fetch_orderbook()
            bal_quote, bal_base = get_balances()

            ts = now_ts()
            if (ts - last_snapshot_ts) >= CFG.snapshot_interval_s:
                mid = mid_price(bid, ask)
                eq = compute_equity_quote(bal_quote, bal_base, mid)
                save_snapshot(con, ts, eq, bal_quote, bal_base, mid)
                last_snapshot_ts = ts

            sp = spread_frac(bid, ask)
            print(
                f"TICK bid={bid:.7f} ask={ask:.7f} spread={sp*100:.4f}% "
                f"bid_xlm={bid_xlm:.2f} ask_xlm={ask_xlm:.2f} "
                f"bal{CFG.quote_asset}={bal_quote:.2f} bal{CFG.base_asset}={bal_base:.6f}"
            )

            action = decide_and_trade(con, bid, ask, bid_xlm, ask_xlm, bal_quote, bal_base)

            # Print report about once per minute and also after trading
            if action == "traded" or (ts % 60 < int(CFG.tick_interval_s)):
                bal_quote, bal_base = get_balances()
                print(report_line(con, bid, ask, bid_xlm, ask_xlm, bal_quote, bal_base))

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

        time.sleep(CFG.tick_interval_s)


if __name__ == "__main__":
    main()


