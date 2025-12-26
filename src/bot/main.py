import os
import time
import math
import sqlite3
import random
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    base_asset: str = os.getenv("BASE_ASSET", "XLM")
    quote_asset: str = os.getenv("QUOTE_ASSET", "USDC")

    # --- Trading frequency / selectivity ---
    # Higher = fewer trades
    min_spread_bps: float = float(os.getenv("MIN_SPREAD_BPS", "60"))  # 60 bps = 0.60%
    trade_cooldown_s: int = int(os.getenv("TRADE_COOLDOWN_S", "240"))  # 4 minutes

    # --- Profit targeting ---
    # Minimum profit you want per round-trip (before fees, in bps)
    min_profit_bps: float = float(os.getenv("MIN_PROFIT_BPS", "25"))  # 0.25%
    # Extra buffer above min profit to reduce trades further
    profit_buffer_bps: float = float(os.getenv("PROFIT_BUFFER_BPS", "10"))  # 0.10%

    # --- Order behavior (simulated maker orders) ---
    # How far from mid to place the buy (bps). Larger -> fewer fills -> fewer trades.
    entry_offset_bps: float = float(os.getenv("ENTRY_OFFSET_BPS", "10"))  # 0.10% below mid
    # Timeout for open orders (seconds)
    order_timeout_s: int = int(os.getenv("ORDER_TIMEOUT_S", "900"))  # 15 minutes

    # Trade size in quote (USDC) used per cycle
    trade_quote_amount: float = float(os.getenv("TRADE_QUOTE_AMOUNT", "15"))
    min_base_lot: float = float(os.getenv("MIN_BASE_LOT", "0.5"))

    # Fees (in bps of quote notional per fill) - set to 0 for pure paper
    fee_bps: float = float(os.getenv("FEE_BPS", "0"))  # e.g. 5 = 0.05%

    # Paper market simulation
    start_price: float = float(os.getenv("START_PRICE", "0.215"))
    sim_vol_bps: float = float(os.getenv("SIM_VOL_BPS", "10"))     # random walk step size (bps)
    sim_spread_bps: float = float(os.getenv("SIM_SPREAD_BPS", "45"))  # typical spread in bps

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


# ============================================================
# SQLITE PERSISTENCE
# ============================================================

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
    con.execute(
        "INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (k, v)
    )
    con.commit()

def log_trade(
    con: sqlite3.Connection,
    ts: int,
    side: str,
    price: float,
    base_qty: float,
    quote_qty: float,
    fee_quote: float = 0.0,
    note: str = ""
) -> None:
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

def last_trade_ts(con: sqlite3.Connection) -> Optional[int]:
    cur = con.execute("SELECT ts FROM trades ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    return int(row[0]) if row else None


# ============================================================
# HELPERS
# ============================================================

def now_ts() -> int:
    return int(time.time())

def bps_from_frac(x: float) -> float:
    return x * 10_000.0

def frac_from_bps(bps: float) -> float:
    return bps / 10_000.0

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

def fee_quote_for_notional(quote_notional: float) -> float:
    return quote_notional * frac_from_bps(CFG.fee_bps)

def should_start_new_cycle(con: sqlite3.Connection, spread_bps: float) -> Tuple[bool, str]:
    if spread_bps < CFG.min_spread_bps:
        return False, f"spread<{CFG.min_spread_bps:.2f}bps"
    lt = last_trade_ts(con)
    if lt is not None and (now_ts() - lt) < CFG.trade_cooldown_s:
        return False, f"cooldown<{CFG.trade_cooldown_s}s"
    return True, "ok"

def quantize_base(x: float) -> float:
    return math.floor(x * 1_000_000) / 1_000_000


# ============================================================
# PAPER MARKET + WALLET
# ============================================================

class PaperMarket:
    def __init__(self, start: float):
        self.mid = start

    def step(self) -> Tuple[float, float, float, float]:
        step_bps = random.uniform(-CFG.sim_vol_bps, CFG.sim_vol_bps)
        self.mid *= (1.0 + step_bps / 10_000.0)

        half_spread_bps = max(1.0, CFG.sim_spread_bps) / 2.0
        bid = self.mid * (1.0 - half_spread_bps / 10_000.0)
        ask = self.mid * (1.0 + half_spread_bps / 10_000.0)

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

def wallet_buy_fill(price: float, base_qty: float) -> bool:
    cost = price * base_qty
    fee = fee_quote_for_notional(cost)
    total_cost = cost + fee
    if total_cost > PAPER_WALLET.quote:
        return False
    PAPER_WALLET.quote -= total_cost
    PAPER_WALLET.base += base_qty
    return True

def wallet_sell_fill(price: float, base_qty: float) -> bool:
    if base_qty > PAPER_WALLET.base:
        return False
    proceeds = price * base_qty
    fee = fee_quote_for_notional(proceeds)
    net = proceeds - fee
    PAPER_WALLET.base -= base_qty
    PAPER_WALLET.quote += net
    return True


# ============================================================
# STRATEGY STATE: maker-like BUY then SELL
# ============================================================

class OpenOrder:
    def __init__(self, side: str, price: float, base_qty: float, created_ts: int):
        self.side = side
        self.price = price
        self.base_qty = base_qty
        self.created_ts = created_ts

class StrategyState:
    def __init__(self):
        self.open_order: Optional[OpenOrder] = None
        self.last_buy_price: Optional[float] = None

STATE = StrategyState()

def maybe_cancel_stale_order() -> Optional[str]:
    if STATE.open_order is None:
        return None
    age = now_ts() - STATE.open_order.created_ts
    if age >= CFG.order_timeout_s:
        side = STATE.open_order.side
        STATE.open_order = None
        return f"cancel_{side.lower()}_timeout"
    return None

def place_new_buy(con: sqlite3.Connection, bid: float, ask: float, bal_quote: float) -> str:
    mid = mid_price(bid, ask)
    sp_bps = bps_from_frac(spread_frac(bid, ask))
    ok, reason = should_start_new_cycle(con, sp_bps)
    if not ok:
        return reason

    desired_quote = min(CFG.trade_quote_amount, bal_quote)
    if desired_quote <= 0:
        return "no_quote_balance"

    buy_price = mid * (1.0 - frac_from_bps(CFG.entry_offset_bps))
    base_qty = quantize_base(desired_quote / buy_price)

    if base_qty < CFG.min_base_lot:
        return "base_qty_too_small"

    STATE.open_order = OpenOrder("BUY", buy_price, base_qty, now_ts())
    return f"place_buy@{buy_price:.7f}"

def try_fill_order(con: sqlite3.Connection, bid: float, ask: float) -> Optional[str]:
    if STATE.open_order is None:
        return None

    o = STATE.open_order
    ts = now_ts()

    if o.side == "BUY":
        # BUY fills if ask <= our limit
        if ask <= o.price:
            cost = o.price * o.base_qty
            fee = fee_quote_for_notional(cost)
            if wallet_buy_fill(o.price, o.base_qty):
                log_trade(con, ts, "BUY", o.price, o.base_qty, cost, fee, "paper_maker")
                STATE.last_buy_price = o.price

                target_bps = CFG.min_profit_bps + CFG.profit_buffer_bps
                sell_price = STATE.last_buy_price * (1.0 + frac_from_bps(target_bps))
                STATE.open_order = OpenOrder("SELL", sell_price, o.base_qty, ts)
                return "buy_filled_sell_placed"
            else:
                STATE.open_order = None
                return "buy_fill_failed"

    elif o.side == "SELL":
        # SELL fills if bid >= our limit
        if bid >= o.price:
            proceeds = o.price * o.base_qty
            fee = fee_quote_for_notional(proceeds)
            if wallet_sell_fill(o.price, o.base_qty):
                log_trade(con, ts, "SELL", o.price, o.base_qty, proceeds, fee, "paper_maker")
                STATE.open_order = None
                STATE.last_buy_price = None
                return "sell_filled_cycle_done"
            else:
                STATE.open_order = None
                return "sell_fill_failed"

    return None

def decide_and_trade(con: sqlite3.Connection, bid: float, ask: float, bid_xlm: float, ask_xlm: float, bal_quote: float, bal_base: float) -> str:
    stale = maybe_cancel_stale_order()
    if stale:
        return stale

    filled = try_fill_order(con, bid, ask)
    if filled:
        return filled

    if STATE.open_order is None:
        return place_new_buy(con, bid, ask, bal_quote)

    return "waiting"


# ============================================================
# REPORTING: realized vs unrealized (FIXED)
# ============================================================

def realized_pnl_total(con: sqlite3.Connection) -> float:
    cur = con.execute(
        "SELECT ts, side, price, base_qty, quote_qty, fee_quote FROM trades ORDER BY ts ASC, id ASC"
    )

    pos_base = 0.0
    cost_quote = 0.0
    realized = 0.0

    for ts, side, price, base_qty, quote_qty, fee_quote in cur.fetchall():
        fee_quote = float(fee_quote or 0.0)
        base_qty = float(base_qty)
        quote_qty = float(quote_qty)

        if side.upper() == "BUY":
            pos_base += base_qty
            cost_quote += (quote_qty + fee_quote)

        elif side.upper() == "SELL":
            if pos_base <= 0:
                continue

            avg_cost = cost_quote / pos_base
            removed_cost = avg_cost * base_qty
            proceeds = quote_qty - fee_quote

            realized += (proceeds - removed_cost)

            pos_base -= base_qty
            cost_quote -= removed_cost

            if pos_base < 1e-12:
                pos_base = 0.0
                cost_quote = 0.0

    return realized

def realized_pnl_since(con: sqlite3.Connection, since_ts: int) -> float:
    cur = con.execute(
        "SELECT ts, side, base_qty, quote_qty, fee_quote FROM trades ORDER BY ts ASC, id ASC"
    )

    pos_base = 0.0
    cost_quote = 0.0
    realized_since = 0.0

    for ts, side, base_qty, quote_qty, fee_quote in cur.fetchall():
        ts = int(ts)
        fee_quote = float(fee_quote or 0.0)
        base_qty = float(base_qty)
        quote_qty = float(quote_qty)

        if side.upper() == "BUY":
            pos_base += base_qty
            cost_quote += (quote_qty + fee_quote)

        elif side.upper() == "SELL":
            if pos_base <= 0:
                continue

            avg_cost = cost_quote / pos_base
            removed_cost = avg_cost * base_qty
            proceeds = quote_qty - fee_quote
            pnl = (proceeds - removed_cost)

            if ts >= since_ts:
                realized_since += pnl

            pos_base -= base_qty
            cost_quote -= removed_cost

            if pos_base < 1e-12:
                pos_base = 0.0
                cost_quote = 0.0

    return realized_since

def report_line(con: sqlite3.Connection, bid: float, ask: float, bid_xlm: float, ask_xlm: float, bal_quote: float, bal_base: float) -> str:
    ts = now_ts()
    mid = mid_price(bid, ask)

    equity_now = compute_equity_quote(bal_quote, bal_base, mid)
    initial = get_or_set_initial_equity(con, equity_now)
    total_pnl = equity_now - initial

    realized_total = realized_pnl_total(con)
    unrealized = total_pnl - realized_total

    t48 = ts - int(48 * 3600)
    realized_48h = realized_pnl_since(con, t48)
    trades_48h = count_trades_since(con, t48)

    ord_txt = "none"
    if STATE.open_order is not None:
        o = STATE.open_order
        ord_txt = f"{o.side}@{o.price:.7f} qty={o.base_qty:.6f} age={ts-o.created_ts}s"

    # ✅ IMPROVEMENT: clearer "profit_*" labels
    return (
        f"REPORT 48h trades={trades_48h} "
        f"profit_48h_realized={realized_48h:.6f} {CFG.quote_asset} | "
        f"equity={equity_now:.2f} {CFG.quote_asset} | "
        f"profit_realized_total={realized_total:.2f} {CFG.quote_asset} | "
        f"profit_unrealized={unrealized:.2f} {CFG.quote_asset} | "
        f"profit_total_now={total_pnl:.2f} {CFG.quote_asset} | "
        f"order={ord_txt} requote_bps={int(CFG.requote_bps)}"
    )


# ============================================================
# MAIN LOOP
# ============================================================

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

            if (
                action in {"buy_filled_sell_placed", "sell_filled_cycle_done"}
                or (ts % 60 < int(CFG.tick_interval_s))
            ):
                bal_quote, bal_base = get_balances()
                print(report_line(con, bid, ask, bid_xlm, ask_xlm, bal_quote, bal_base))

            if action not in {"waiting"} and not action.startswith("cooldown") and not action.startswith("spread<"):
                print(f"ACTION: {action}")

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

        time.sleep(CFG.tick_interval_s)


if __name__ == "__main__":
    main()
