#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Any, Dict

import ccxt


# -------------------------
# Helpers
# -------------------------

def getenv_str(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default

def getenv_int(key: str, default: int) -> int:
    v = getenv_str(key, None)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default

def getenv_float(key: str, default: float) -> float:
    v = getenv_str(key, None)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default

def getenv_bool(key: str, default: bool = False) -> bool:
    v = getenv_str(key, None)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    exchange_id: str
    api_key: Optional[str]
    api_secret: Optional[str]
    api_password: Optional[str]
    use_testnet: bool

    symbol: str                  # e.g. "XRP/USDT"
    base_asset: str              # derived from symbol
    quote_asset: str             # derived from symbol

    min_spread_bps: float
    requote_bps: float
    trade_cooldown_s: int
    trade_quote_amount: float
    min_base_lot: float

    tick_interval_s: int
    snapshot_interval_s: int

    max_open_orders: int
    order_ttl_s: int

    take_profit_bps: float
    max_hold_s: int
    min_exit_profit_bps: float

    db_path: str

    def validate(self) -> None:
        if "/" not in self.symbol:
            raise ValueError('SYMBOL must look like "XRP/USDT"')
        if not self.api_key or not self.api_secret:
            raise ValueError("API_KEY and API_SECRET are required for LIVE trading.")
        if self.trade_quote_amount <= 0:
            raise ValueError("TRADE_QUOTE_AMOUNT must be > 0")
        if self.min_spread_bps <= 0:
            raise ValueError("MIN_SPREAD_BPS must be > 0")
        if self.tick_interval_s <= 0:
            raise ValueError("TICK_INTERVAL_S must be > 0")
        if self.snapshot_interval_s <= 0:
            raise ValueError("SNAPSHOT_INTERVAL_S must be > 0")
        if self.requote_bps < 0:
            raise ValueError("REQUOTE_BPS must be >= 0")
        if self.take_profit_bps < 0:
            raise ValueError("TAKE_PROFIT_BPS must be >= 0")
        if self.max_hold_s < 0:
            raise ValueError("MAX_HOLD_S must be >= 0")
        if self.min_exit_profit_bps < 0:
            raise ValueError("MIN_EXIT_PROFIT_BPS must be >= 0")


def print_env_check() -> None:
    keys = [
        "EXCHANGE","API_KEY","API_SECRET","API_PASSWORD","USE_TESTNET",
        "SYMBOL",
        "MIN_SPREAD_BPS","REQUOTE_BPS","TRADE_COOLDOWN_S","TRADE_QUOTE_AMOUNT","MIN_BASE_LOT",
        "TICK_INTERVAL_S","SNAPSHOT_INTERVAL_S",
        "DB_PATH","MAX_OPEN_ORDERS","ORDER_TTL_S",
        "TAKE_PROFIT_BPS","MAX_HOLD_S","MIN_EXIT_PROFIT_BPS",
    ]
    print("=== ENV CHECK ===")
    for k in keys:
        v = os.getenv(k)
        if k in ("API_KEY","API_SECRET","API_PASSWORD") and v:
            v = v[:3] + "..." + v[-3:]
        print(f"{k}={v!r}")
    print("=================")


def load_config() -> Config:
    symbol = (getenv_str("SYMBOL", "XRP/USDT") or "XRP/USDT").upper()
    base, quote = symbol.split("/", 1)

    cfg = Config(
        exchange_id=(getenv_str("EXCHANGE", "binance") or "binance").lower(),
        api_key=getenv_str("API_KEY", None),
        api_secret=getenv_str("API_SECRET", None),
        api_password=getenv_str("API_PASSWORD", None),
        use_testnet=getenv_bool("USE_TESTNET", False),

        symbol=symbol,
        base_asset=base,
        quote_asset=quote,

        min_spread_bps=getenv_float("MIN_SPREAD_BPS", 35.0),
        requote_bps=getenv_float("REQUOTE_BPS", 10.0),
        trade_cooldown_s=getenv_int("TRADE_COOLDOWN_S", 600),
        trade_quote_amount=getenv_float("TRADE_QUOTE_AMOUNT", 25.0),
        min_base_lot=getenv_float("MIN_BASE_LOT", 5.0),

        tick_interval_s=getenv_int("TICK_INTERVAL_S", 2),
        snapshot_interval_s=getenv_int("SNAPSHOT_INTERVAL_S", 30),

        max_open_orders=getenv_int("MAX_OPEN_ORDERS", 1),
        order_ttl_s=getenv_int("ORDER_TTL_S", 900),

        take_profit_bps=getenv_float("TAKE_PROFIT_BPS", 12.0),
        max_hold_s=getenv_int("MAX_HOLD_S", 20 * 60),
        min_exit_profit_bps=getenv_float("MIN_EXIT_PROFIT_BPS", 2.0),

        db_path=getenv_str("DB_PATH", "/tmp/bot.db") or "/tmp/bot.db",
    )
    cfg.validate()
    return cfg


# -------------------------
# DB (unchanged)
# -------------------------

class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          side TEXT NOT NULL,
          price REAL NOT NULL,
          base_qty REAL NOT NULL,
          quote_qty REAL NOT NULL,
          realized_pnl_quote REAL NOT NULL DEFAULT 0.0,
          mode TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          bid REAL NOT NULL,
          ask REAL NOT NULL,
          spread_bps REAL NOT NULL,
          base_bal REAL NOT NULL,
          quote_bal REAL NOT NULL,
          equity_quote REAL NOT NULL,
          unrealized_quote REAL NOT NULL,
          realized_total_quote REAL NOT NULL,
          open_order_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS state (
          k TEXT PRIMARY KEY,
          v TEXT NOT NULL
        );
        """)
        self.conn.commit()

    def set_state(self, k: str, v: Any) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (k, json.dumps(v)),
        )
        self.conn.commit()

    def get_state(self, k: str, default: Any = None) -> Any:
        cur = self.conn.cursor()
        cur.execute("SELECT v FROM state WHERE k=?", (k,))
        row = cur.fetchone()
        if not row:
            return default
        try:
            return json.loads(row["v"])
        except Exception:
            return default

    def add_trade(self, ts: datetime, side: str, price: float, base_qty: float, quote_qty: float,
                  realized_pnl_quote: float, mode: str) -> None:
        cur = self.conn.cursor()
        cur.execute("""
          INSERT INTO trades(ts,side,price,base_qty,quote_qty,realized_pnl_quote,mode)
          VALUES(?,?,?,?,?,?,?)
        """, (ts.isoformat(), side, price, base_qty, quote_qty, realized_pnl_quote, mode))
        self.conn.commit()

    def count_trades_since(self, since: datetime) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM trades WHERE ts >= ?", (since.isoformat(),))
        return int(cur.fetchone()["c"])

    def realized_pnl_since(self, since: datetime) -> float:
        cur = self.conn.cursor()
        cur.execute("SELECT COALESCE(SUM(realized_pnl_quote),0) AS s FROM trades WHERE ts >= ?",
                    (since.isoformat(),))
        return float(cur.fetchone()["s"])

    def realized_pnl_total(self) -> float:
        cur = self.conn.cursor()
        cur.execute("SELECT COALESCE(SUM(realized_pnl_quote),0) AS s FROM trades")
        return float(cur.fetchone()["s"])

    def add_snapshot(self, **kwargs) -> None:
        cur = self.conn.cursor()
        cur.execute("""
          INSERT INTO snapshots(ts,bid,ask,spread_bps,base_bal,quote_bal,equity_quote,unrealized_quote,realized_total_quote,open_order_json)
          VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
            kwargs["ts"].isoformat(),
            kwargs["bid"], kwargs["ask"], kwargs["spread_bps"],
            kwargs["base_bal"], kwargs["quote_bal"],
            kwargs["equity_quote"], kwargs["unrealized_quote"], kwargs["realized_total_quote"],
            kwargs.get("open_order_json"),
        ))
        self.conn.commit()


# -------------------------
# Exchange (ccxt)
# -------------------------

def make_exchange(cfg: Config) -> ccxt.Exchange:
    if not hasattr(ccxt, cfg.exchange_id):
        raise ValueError(f"Unknown exchange id: {cfg.exchange_id}")

    klass = getattr(ccxt, cfg.exchange_id)
    ex = klass({
        "apiKey": cfg.api_key,
        "secret": cfg.api_secret,
        "password": cfg.api_password or None,
        "enableRateLimit": True,
        "options": {},
    })

    # Testnet support (primarily Binance here; others differ)
    if cfg.exchange_id == "binance" and cfg.use_testnet:
        ex.set_sandbox_mode(True)

    ex.load_markets()
    if cfg.symbol not in ex.markets:
        raise ValueError(f"Symbol {cfg.symbol} not found on {cfg.exchange_id}.")
    return ex

def fetch_best_bid_ask(ex: ccxt.Exchange, symbol: str) -> Tuple[float, float]:
    t = ex.fetch_ticker(symbol)
    bid = safe_float(t.get("bid"), 0.0)
    ask = safe_float(t.get("ask"), 0.0)
    if bid <= 0 or ask <= 0:
        # fallback to orderbook
        ob = ex.fetch_order_book(symbol, limit=5)
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            raise RuntimeError("Orderbook empty.")
        bid = safe_float(bids[0][0], 0.0)
        ask = safe_float(asks[0][0], 0.0)
    if bid <= 0 or ask <= 0:
        raise RuntimeError("Invalid bid/ask.")
    return bid, ask

def fetch_balances(ex: ccxt.Exchange, base: str, quote: str) -> Tuple[float, float]:
    b = ex.fetch_balance()
    base_free = safe_float(((b.get(base) or {}).get("free")), 0.0)
    quote_free = safe_float(((b.get(quote) or {}).get("free")), 0.0)
    return base_free, quote_free

def round_amount_to_market(ex: ccxt.Exchange, symbol: str, amount: float) -> float:
    # ccxt helper (respects step size)
    return safe_float(ex.amount_to_precision(symbol, amount), 0.0)

def round_price_to_market(ex: ccxt.Exchange, symbol: str, price: float) -> float:
    return safe_float(ex.price_to_precision(symbol, price), 0.0)


# -------------------------
# Bot
# -------------------------

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(cfg.db_path)
        self.ex = make_exchange(cfg)

        self.last_trade_ts = self._load_last_trade_ts()
        self.open_order = self.db.get_state("open_order", default=None)
        self.last_snapshot_ts = 0.0

        # Position state (cost basis)
        self.position = self.db.get_state("position", default=None)

        print("=== CONFIG RESOLVED ===")
        d = asdict(cfg)
        if d.get("api_key"):
            d["api_key"] = d["api_key"][:3] + "..." + d["api_key"][-3:]
        if d.get("api_secret"):
            d["api_secret"] = d["api_secret"][:3] + "..." + d["api_secret"][-3:]
        if d.get("api_password"):
            d["api_password"] = d["api_password"][:3] + "..." + d["api_password"][-3:]
        print(json.dumps(d, indent=2))
        print("=======================")
        print("MODE: LIVE (exchange)")

    # -------------------------
    # State helpers
    # -------------------------

    def _load_last_trade_ts(self) -> float:
        v = self.db.get_state("last_trade_ts", default=0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    def _set_last_trade_ts(self, t: float) -> None:
        self.last_trade_ts = t
        self.db.set_state("last_trade_ts", t)

    def _set_open_order(self, order: Optional[dict]) -> None:
        self.open_order = order
        self.db.set_state("open_order", order)

    def _set_position(self, pos: Optional[dict]) -> None:
        self.position = pos
        self.db.set_state("position", pos)

    def _cooldown_ok(self) -> bool:
        return (time.time() - self.last_trade_ts) >= self.cfg.trade_cooldown_s

    # -------------------------
    # Math / reporting
    # -------------------------

    def _spread_bps(self, bid: float, ask: float) -> float:
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid) * 10000.0

    def _equity_quote(self, bid: float, base_bal: float, quote_bal: float) -> float:
        return quote_bal + base_bal * bid

    def _position_unrealized(self, bid: float) -> float:
        if not self.position:
            return 0.0
        base_qty = float(self.position.get("base_qty", 0.0))
        entry_cost = float(self.position.get("entry_cost_quote", 0.0))
        return (base_qty * bid) - entry_cost

    def _report(self, bid: float, ask: float, base_bal: float, quote_bal: float, open_order_str: str) -> None:
        since = now_utc() - timedelta(hours=48)
        trades_48h = self.db.count_trades_since(since)
        realized_48h = self.db.realized_pnl_since(since)
        realized_total = self.db.realized_pnl_total()

        equity = self._equity_quote(bid, base_bal, quote_bal)
        unreal = self._position_unrealized(bid)
        total_now = realized_total + unreal

        print(
            f"REPORT 48h trades={trades_48h} "
            f"profit_48h_realized={realized_48h:.6f} {self.cfg.quote_asset} | "
            f"equity={equity:.2f} {self.cfg.quote_asset} | "
            f"profit_realized_total={realized_total:.2f} {self.cfg.quote_asset} | "
            f"profit_unrealized={unreal:.2f} {self.cfg.quote_asset} | "
            f"profit_total_now={total_now:.2f} {self.cfg.quote_asset} | "
            f"order={open_order_str} requote_bps={self.cfg.requote_bps:g}"
        )

    def _maybe_snapshot(self, bid: float, ask: float, spread_bps: float, base_bal: float, quote_bal: float) -> None:
        now = time.time()
        if now - self.last_snapshot_ts < self.cfg.snapshot_interval_s:
            return
        self.last_snapshot_ts = now

        equity = self._equity_quote(bid, base_bal, quote_bal)
        unreal = self._position_unrealized(bid)
        realized_total = self.db.realized_pnl_total()

        self.db.add_snapshot(
            ts=now_utc(),
            bid=bid, ask=ask,
            spread_bps=spread_bps,
            base_bal=base_bal,
            quote_bal=quote_bal,
            equity_quote=equity,
            unrealized_quote=unreal,
            realized_total_quote=realized_total,
            open_order_json=json.dumps(self.open_order) if self.open_order else None
        )

    # -------------------------
    # Profit/exit decision
    # -------------------------

    def _pos_gain_bps(self, bid: float) -> float:
        if not self.position:
            return 0.0
        entry_price = float(self.position.get("entry_price", 0.0))
        if entry_price <= 0:
            return 0.0
        return ((bid - entry_price) / entry_price) * 10000.0

    def _pos_age_s(self) -> int:
        if not self.position:
            return 0
        ts = float(self.position.get("entry_ts", time.time()))
        return int(time.time() - ts)

    def _should_force_exit(self, bid: float) -> Tuple[bool, str]:
        if not self.position:
            return False, ""
        gain_bps = self._pos_gain_bps(bid)
        age_s = self._pos_age_s()

        if gain_bps >= self.cfg.take_profit_bps:
            return True, f"take_profit gain={gain_bps:.2f}bps"
        if self.cfg.max_hold_s > 0 and age_s >= self.cfg.max_hold_s and gain_bps >= self.cfg.min_exit_profit_bps:
            return True, f"max_hold age={age_s}s gain={gain_bps:.2f}bps"
        return False, ""

    # -------------------------
    # Position + realized pnl accounting
    # -------------------------

    def _apply_buy_to_position(self, ts: datetime, filled_base: float, spent_quote: float, avg_price: float) -> None:
        if filled_base <= 0 or spent_quote <= 0:
            return

        if not self.position:
            self._set_position({
                "entry_ts": time.time(),
                "entry_price": float(avg_price),
                "base_qty": float(filled_base),
                "entry_cost_quote": float(spent_quote),
            })
        else:
            base_qty = float(self.position.get("base_qty", 0.0))
            entry_cost = float(self.position.get("entry_cost_quote", 0.0))

            new_base = base_qty + filled_base
            new_cost = entry_cost + spent_quote
            new_entry_price = (new_cost / new_base) if new_base > 0 else float(avg_price)

            self.position["base_qty"] = float(new_base)
            self.position["entry_cost_quote"] = float(new_cost)
            self.position["entry_price"] = float(new_entry_price)
            self._set_position(self.position)

        self.db.add_trade(ts, "BUY", float(avg_price), float(filled_base), float(spent_quote), 0.0, "LIVE")

    def _apply_sell_from_position(self, ts: datetime, sold_base: float, received_quote: float, avg_price: float) -> float:
        if sold_base <= 0 or received_quote <= 0:
            return 0.0

        realized = 0.0
        if self.position:
            base_qty = float(self.position.get("base_qty", 0.0))
            entry_cost = float(self.position.get("entry_cost_quote", 0.0))

            if base_qty > 0 and entry_cost >= 0:
                sell_qty = min(sold_base, base_qty)
                cost_per_base = entry_cost / base_qty if base_qty > 0 else 0.0
                cost_basis = cost_per_base * sell_qty

                realized = received_quote - cost_basis

                remaining_base = base_qty - sell_qty
                remaining_cost = max(0.0, entry_cost - cost_basis)

                if remaining_base <= 1e-9:
                    self._set_position(None)
                else:
                    new_entry_price = remaining_cost / remaining_base if remaining_base > 0 else float(avg_price)
                    self.position["base_qty"] = float(remaining_base)
                    self.position["entry_cost_quote"] = float(remaining_cost)
                    self.position["entry_price"] = float(new_entry_price)
                    self._set_position(self.position)

        self.db.add_trade(ts, "SELL", float(avg_price), float(sold_base), float(received_quote), float(realized), "LIVE")
        return realized

    # -------------------------
    # Order placement + fill handling (simplified)
    # -------------------------

    def _place_limit_buy_and_wait(self, amount_base: float, price: float) -> Tuple[float, float, float]:
        """
        Places a LIMIT BUY for base, waits for fill (or cancels on TTL).
        Returns (filled_base, spent_quote, avg_price).
        """
        symbol = self.cfg.symbol
        amount_base = round_amount_to_market(self.ex, symbol, amount_base)
        price = round_price_to_market(self.ex, symbol, price)
        if amount_base <= 0 or price <= 0:
            return 0.0, 0.0, 0.0

        order = self.ex.create_limit_buy_order(symbol, amount_base, price)
        oid = order["id"]
        t_start = time.time()

        while True:
            o = self.ex.fetch_order(oid, symbol)
            status = (o.get("status") or "").lower()
            filled = safe_float(o.get("filled"), 0.0)
            cost = safe_float(o.get("cost"), 0.0)  # quote spent
            avg = safe_float(o.get("average"), 0.0) or (cost / filled if filled > 0 else 0.0)

            if status in ("closed",) or (filled > 0 and safe_float(o.get("remaining"), 0.0) == 0.0):
                return filled, cost, avg

            if time.time() - t_start > self.cfg.order_ttl_s:
                try:
                    self.ex.cancel_order(oid, symbol)
                except Exception:
                    pass
                # re-fetch for partial fill info
                o = self.ex.fetch_order(oid, symbol)
                filled = safe_float(o.get("filled"), 0.0)
                cost = safe_float(o.get("cost"), 0.0)
                avg = safe_float(o.get("average"), 0.0) or (cost / filled if filled > 0 else 0.0)
                return filled, cost, avg

            time.sleep(0.5)

    def _place_limit_sell_and_wait(self, amount_base: float, price: float) -> Tuple[float, float, float]:
        """
        Places a LIMIT SELL for base, waits for fill (or cancels on TTL).
        Returns (sold_base, received_quote, avg_price).
        """
        symbol = self.cfg.symbol
        amount_base = round_amount_to_market(self.ex, symbol, amount_base)
        price = round_price_to_market(self.ex, symbol, price)
        if amount_base <= 0 or price <= 0:
            return 0.0, 0.0, 0.0

        order = self.ex.create_limit_sell_order(symbol, amount_base, price)
        oid = order["id"]
        t_start = time.time()

        while True:
            o = self.ex.fetch_order(oid, symbol)
            status = (o.get("status") or "").lower()
            filled = safe_float(o.get("filled"), 0.0)  # base sold
            cost = safe_float(o.get("cost"), 0.0)      # quote received (for sells, cost is usually quote proceeds)
            avg = safe_float(o.get("average"), 0.0) or (cost / filled if filled > 0 else 0.0)

            if status in ("closed",) or (filled > 0 and safe_float(o.get("remaining"), 0.0) == 0.0):
                return filled, cost, avg

            if time.time() - t_start > self.cfg.order_ttl_s:
                try:
                    self.ex.cancel_order(oid, symbol)
                except Exception:
                    pass
                o = self.ex.fetch_order(oid, symbol)
                filled = safe_float(o.get("filled"), 0.0)
                cost = safe_float(o.get("cost"), 0.0)
                avg = safe_float(o.get("average"), 0.0) or (cost / filled if filled > 0 else 0.0)
                return filled, cost, avg

            time.sleep(0.5)

    # -------------------------
    # Trading logic
    # -------------------------

    def _trade(self, bid: float, ask: float, base_bal: float, quote_bal: float) -> None:
        # 1) Exit logic (realize profit)
        force_exit, reason = self._should_force_exit(bid)
        if self.position and force_exit and self._cooldown_ok():
            pos_base = float(self.position.get("base_qty", 0.0))
            sell_qty = min(pos_base, base_bal)
            if sell_qty >= self.cfg.min_base_lot:
                sold_base, got_quote, avg = self._place_limit_sell_and_wait(sell_qty, bid)
                if sold_base > 0 and got_quote > 0:
                    realized = self._apply_sell_from_position(now_utc(), sold_base, got_quote, avg)
                    print(f"REALIZED = {realized:.6f} {self.cfg.quote_asset} ({reason})")
                    self._set_last_trade_ts(time.time())
            return

        # 2) Entry logic (spread gate)
        if not self._cooldown_ok():
            return

        spread_bps = self._spread_bps(bid, ask)
        if spread_bps < self.cfg.min_spread_bps:
            return

        # Only buy if no position (same as your code)
        if (not self.position) and quote_bal >= self.cfg.trade_quote_amount:
            buy_price = ask
            base_qty = self.cfg.trade_quote_amount / max(buy_price, 1e-12)
            if base_qty < self.cfg.min_base_lot:
                return

            filled_base, spent_quote, avg = self._place_limit_buy_and_wait(base_qty, buy_price)
            if filled_base > 0 and spent_quote > 0:
                self._apply_buy_to_position(now_utc(), filled_base, spent_quote, avg)
                self._set_last_trade_ts(time.time())

    # -------------------------
    # Loop
    # -------------------------

    def run(self) -> None:
        print("Starting bot loop...")
        while True:
            t0 = time.time()
            try:
                bid, ask = fetch_best_bid_ask(self.ex, self.cfg.symbol)
                spread_bps = self._spread_bps(bid, ask)

                base_bal, quote_bal = fetch_balances(self.ex, self.cfg.base_asset, self.cfg.quote_asset)

                self._trade(bid, ask, base_bal, quote_bal)

                # refresh balances after trade
                base_bal, quote_bal = fetch_balances(self.ex, self.cfg.base_asset, self.cfg.quote_asset)

                self._maybe_snapshot(bid, ask, spread_bps, base_bal, quote_bal)

                print(
                    f"TICK bid={bid:.6f} ask={ask:.6f} "
                    f"spread={spread_bps/100:.4f}% "
                    f"bal{self.cfg.quote_asset}={quote_bal:.2f} "
                    f"bal{self.cfg.base_asset}={base_bal:.6f}"
                )

                if int(time.time()) % max(self.cfg.snapshot_interval_s, 1) == 0:
                    self._report(bid, ask, base_bal, quote_bal, "none")

            except Exception as e:
                print("ERROR:", str(e))

            elapsed = time.time() - t0
            time.sleep(max(0.0, self.cfg.tick_interval_s - elapsed))


def main():
    print_env_check()
    cfg = load_config()
    bot = Bot(cfg)
    bot.run()


if __name__ == "__main__":
    main()
