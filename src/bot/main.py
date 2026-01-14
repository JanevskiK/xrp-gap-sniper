#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Any, Dict

from stellar_sdk import Server, Keypair, TransactionBuilder, Network, Asset
from stellar_sdk.exceptions import BadRequestError


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
    base_asset: str
    quote_asset: str
    base_issuer: Optional[str]
    quote_issuer: Optional[str]

    min_spread_bps: float
    requote_bps: float
    trade_cooldown_s: int
    trade_quote_amount: float
    min_base_lot: float

    tick_interval_s: int
    snapshot_interval_s: int

    start_price: float
    sim_vol_bps: float
    sim_spread_bps: float

    starting_quote: float
    starting_base: float

    horizon_url: str
    secret_key: Optional[str]
    db_path: str

    max_open_orders: int
    order_ttl_s: int

    # ✅ Improvement: realize profit rules
    take_profit_bps: float          # e.g. 12 bps = 0.12%
    max_hold_s: int                 # force exit after this time if profitable enough
    min_exit_profit_bps: float      # exit-on-timeout only if >= this bps

    @property
    def is_live(self) -> bool:
        return bool(self.secret_key and self.secret_key.startswith("S"))

    def validate(self) -> None:
        if self.quote_asset.upper() not in ("XLM", "NATIVE"):
            if not self.quote_issuer:
                raise ValueError("QUOTE_ISSUER is required for non-native QUOTE_ASSET (e.g., USDC).")
        if self.base_asset.upper() not in ("XLM", "NATIVE"):
            if not self.base_issuer:
                raise ValueError("BASE_ISSUER is required for non-native BASE_ASSET.")

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
        "HORIZON_URL",
        "BASE_ASSET","QUOTE_ASSET","BASE_ISSUER","QUOTE_ISSUER",
        "MIN_SPREAD_BPS","REQUOTE_BPS","TRADE_COOLDOWN_S","TRADE_QUOTE_AMOUNT","MIN_BASE_LOT",
        "START_PRICE","SIM_VOL_BPS","SIM_SPREAD_BPS",
        "TICK_INTERVAL_S","SNAPSHOT_INTERVAL_S",
        "STARTING_QUOTE","STARTING_BASE",
        "DB_PATH","MAX_OPEN_ORDERS","ORDER_TTL_S",
        # ✅ Improvement envs
        "TAKE_PROFIT_BPS","MAX_HOLD_S","MIN_EXIT_PROFIT_BPS",
        "SECRET_KEY",
    ]
    print("=== ENV CHECK ===")
    for k in keys:
        v = os.getenv(k)
        if k == "SECRET_KEY" and v:
            v = v[:5] + "..." + v[-5:]
        print(f"{k}={v!r}")
    print("=================")


def load_config() -> Config:
    cfg = Config(
        base_asset=getenv_str("BASE_ASSET", "XLM").upper(),
        quote_asset=getenv_str("QUOTE_ASSET", "USDC").upper(),
        base_issuer=getenv_str("BASE_ISSUER", None),
        quote_issuer=getenv_str("QUOTE_ISSUER", None),

        min_spread_bps=getenv_float("MIN_SPREAD_BPS", 35.0),
        requote_bps=getenv_float("REQUOTE_BPS", 10.0),
        trade_cooldown_s=getenv_int("TRADE_COOLDOWN_S", 600),
        trade_quote_amount=getenv_float("TRADE_QUOTE_AMOUNT", 25.0),
        min_base_lot=getenv_float("MIN_BASE_LOT", 1.0),

        tick_interval_s=getenv_int("TICK_INTERVAL_S", 2),
        snapshot_interval_s=getenv_int("SNAPSHOT_INTERVAL_S", 30),

        start_price=getenv_float("START_PRICE", 0.215),
        sim_vol_bps=getenv_float("SIM_VOL_BPS", 8.0),
        sim_spread_bps=getenv_float("SIM_SPREAD_BPS", 45.0),

        starting_quote=getenv_float("STARTING_QUOTE", 1000.0),
        starting_base=getenv_float("STARTING_BASE", 500.0),

        horizon_url=getenv_str("HORIZON_URL", "https://horizon.stellar.org"),
        secret_key=getenv_str("SECRET_KEY", None),
        db_path=getenv_str("DB_PATH", "/tmp/bot.db"),

        max_open_orders=getenv_int("MAX_OPEN_ORDERS", 2),
        order_ttl_s=getenv_int("ORDER_TTL_S", 900),

        # ✅ Improvement defaults: will realize profit more often
        take_profit_bps=getenv_float("TAKE_PROFIT_BPS", 12.0),          # 0.12%
        max_hold_s=getenv_int("MAX_HOLD_S", 20 * 60),                   # 20min
        min_exit_profit_bps=getenv_float("MIN_EXIT_PROFIT_BPS", 2.0),   # 0.02%
    )
    cfg.validate()
    return cfg


# -------------------------
# DB
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
# Stellar
# -------------------------

def make_asset(code: str, issuer: Optional[str]) -> Asset:
    if code.upper() in ("XLM", "NATIVE"):
        return Asset.native()
    if not issuer:
        raise ValueError(f"Issuer required for asset {code}")
    return Asset(code.upper(), issuer)

def fetch_best_bid_ask(server: Server, base: Asset, quote: Asset) -> Tuple[float, float]:
    ob = server.orderbook(selling=base, buying=quote).call()
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        raise RuntimeError("Orderbook empty (no bids/asks).")
    best_bid = safe_float(bids[0].get("price"), 0.0)
    best_ask = safe_float(asks[0].get("price"), 0.0)
    if best_bid <= 0 or best_ask <= 0:
        raise RuntimeError("Invalid bid/ask from orderbook.")
    return best_bid, best_ask

def fetch_balances(server: Server, cfg: Config) -> Tuple[float, float]:
    kp = Keypair.from_secret(cfg.secret_key)  # type: ignore[arg-type]
    acc = server.accounts().account_id(kp.public_key).call()

    def bal_for(code: str, issuer: Optional[str]) -> float:
        if code.upper() in ("XLM", "NATIVE"):
            for b in acc["balances"]:
                if b["asset_type"] == "native":
                    return safe_float(b["balance"])
            return 0.0
        for b in acc["balances"]:
            if b.get("asset_code") == code.upper() and b.get("asset_issuer") == issuer:
                return safe_float(b["balance"])
        return 0.0

    return bal_for(cfg.base_asset, cfg.base_issuer), bal_for(cfg.quote_asset, cfg.quote_issuer)


# -------------------------
# Bot
# -------------------------

class Bot:
    """
    ✅ Improvement added:
    - Tracks a "position" (entry price/cost) so SELL produces realized profit
    - Uses TAKE_PROFIT_BPS + MAX_HOLD_S (+ MIN_EXIT_PROFIT_BPS) to force realizing profit
    - In LIVE mode, detects fills by balance deltas and records realized profit into DB
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(cfg.db_path)
        self.server = Server(horizon_url=cfg.horizon_url)

        self.base_asset = make_asset(cfg.base_asset, cfg.base_issuer)
        self.quote_asset = make_asset(cfg.quote_asset, cfg.quote_issuer)

        self.last_trade_ts = self._load_last_trade_ts()
        self.open_order = self.db.get_state("open_order", default=None)
        self.last_snapshot_ts = 0.0

        self.sim_base = float(cfg.starting_base)
        self.sim_quote = float(cfg.starting_quote)

        # ✅ Improvement: position state (cost basis)
        self.position = self.db.get_state("position", default=None)  # dict or None

        # ✅ Improvement: last balances (for LIVE fill detection)
        self.last_bal = self.db.get_state("last_bal", default=None)  # {"base":..., "quote":..., "ts":...}

        print("=== CONFIG RESOLVED ===")
        d = asdict(cfg)
        if d.get("secret_key"):
            d["secret_key"] = (d["secret_key"][:5] + "..." + d["secret_key"][-5:])
        print(json.dumps(d, indent=2))
        print("=======================")
        print(f"MODE: {'LIVE' if cfg.is_live else 'SIM'}")

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

    def _set_last_bal(self, base_bal: float, quote_bal: float) -> None:
        self.last_bal = {"base": float(base_bal), "quote": float(quote_bal), "ts": time.time()}
        self.db.set_state("last_bal", self.last_bal)

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
        """
        ✅ Improvement: unrealized is ONLY the open position PnL (paper profit),
        not "equity minus start".
        """
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
        """
        ✅ Improvement: force realizing profit.
        - take-profit if gain_bps >= TAKE_PROFIT_BPS
        - max-hold exit if age >= MAX_HOLD_S AND gain_bps >= MIN_EXIT_PROFIT_BPS
        """
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
    # Offer submit
    # -------------------------

    def _submit_sell_offer(self, base_qty: float, price: float) -> None:
        kp = Keypair.from_secret(self.cfg.secret_key)  # type: ignore[arg-type]
        account = self.server.load_account(kp.public_key)

        tx = (
            TransactionBuilder(
                source_account=account,
                network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,
                base_fee=100
            )
            .append_manage_sell_offer_op(
                selling=self.base_asset,
                buying=self.quote_asset,
                amount=str(round(base_qty, 7)),
                price=str(price),
                offer_id=0
            )
            .set_timeout(30)
            .build()
        )
        tx.sign(kp)
        try:
            self.server.submit_transaction(tx)
        except BadRequestError as e:
            print("LIVE submit SELL offer failed:", str(e)[:350])
            raise

    def _submit_buy_offer(self, base_qty: float, price: float) -> None:
        kp = Keypair.from_secret(self.cfg.secret_key)  # type: ignore[arg-type]
        account = self.server.load_account(kp.public_key)

        tx = (
            TransactionBuilder(
                source_account=account,
                network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,
                base_fee=100
            )
            .append_manage_buy_offer_op(
                selling=self.quote_asset,
                buying=self.base_asset,
                buy_amount=str(round(base_qty, 7)),
                price=str(price),
                offer_id=0
            )
            .set_timeout(30)
            .build()
        )
        tx.sign(kp)
        try:
            self.server.submit_transaction(tx)
        except BadRequestError as e:
            print("LIVE submit BUY offer failed:", str(e)[:350])
            raise

    # -------------------------
    # Position + realized pnl accounting
    # -------------------------

    def _apply_buy_to_position(self, ts: datetime, filled_base: float, spent_quote: float, avg_price: float, mode: str) -> None:
        """
        Update position cost basis.
        """
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
            # Weighted average cost basis (add)
            base_qty = float(self.position.get("base_qty", 0.0))
            entry_cost = float(self.position.get("entry_cost_quote", 0.0))

            new_base = base_qty + filled_base
            new_cost = entry_cost + spent_quote
            new_entry_price = (new_cost / new_base) if new_base > 0 else float(avg_price)

            # keep original entry_ts (or reset; your call). Here we keep first entry_ts.
            self.position["base_qty"] = float(new_base)
            self.position["entry_cost_quote"] = float(new_cost)
            self.position["entry_price"] = float(new_entry_price)
            self._set_position(self.position)

        # trade record (realized_pnl_quote=0 for BUY)
        self.db.add_trade(ts, "BUY", float(avg_price), float(filled_base), float(spent_quote), 0.0, mode)

    def _apply_sell_from_position(self, ts: datetime, sold_base: float, received_quote: float, avg_price: float, mode: str) -> float:
        """
        Reduce/close position and compute realized pnl on the sold portion.
        Returns realized pnl quote (>= or <= 0).
        """
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
                    # keep average entry price consistent with remaining cost
                    new_entry_price = remaining_cost / remaining_base if remaining_base > 0 else float(avg_price)
                    self.position["base_qty"] = float(remaining_base)
                    self.position["entry_cost_quote"] = float(remaining_cost)
                    self.position["entry_price"] = float(new_entry_price)
                    # keep entry_ts as-is
                    self._set_position(self.position)

        # trade record includes realized pnl (only meaningful when it reduces a position)
        self.db.add_trade(ts, "SELL", float(avg_price), float(sold_base), float(received_quote), float(realized), mode)
        return realized

    # -------------------------
    # LIVE: detect fills via balance deltas
    # -------------------------

    def _sync_live_fills_from_balances(self, base_bal: float, quote_bal: float) -> None:
        """
        ✅ Improvement: If we have an open_order, infer a fill by checking balance changes
        since the last loop and then write the trade + realized pnl.

        This is "best effort" but works well for fully filled offers.
        """
        if not self.open_order:
            self._set_last_bal(base_bal, quote_bal)
            return

        prev = self.last_bal or {"base": base_bal, "quote": quote_bal}
        prev_base = float(prev.get("base", base_bal))
        prev_quote = float(prev.get("quote", quote_bal))

        d_base = base_bal - prev_base
        d_quote = quote_bal - prev_quote

        side = self.open_order.get("side")
        # Small epsilon to avoid noise
        eps = 1e-7

        filled = False
        ts = now_utc()

        if side == "BUY":
            # BUY: base increases, quote decreases
            got_base = d_base
            spent_quote = -d_quote
            if got_base > eps and spent_quote > eps:
                avg_price = spent_quote / got_base
                self._apply_buy_to_position(ts, got_base, spent_quote, avg_price, "LIVE")
                filled = True

        elif side == "SELL":
            # SELL: base decreases, quote increases
            sold_base = -d_base
            got_quote = d_quote
            if sold_base > eps and got_quote > eps:
                avg_price = got_quote / sold_base
                realized = self._apply_sell_from_position(ts, sold_base, got_quote, avg_price, "LIVE")
                filled = True
                print(f"REALIZED (LIVE) = {realized:.6f} {self.cfg.quote_asset}")

        if filled:
            # Close the open_order state if it filled
            self._set_open_order(None)

        # Always update last balances
        self._set_last_bal(base_bal, quote_bal)

    # -------------------------
    # Trading logic
    # -------------------------

    def _trade_live(self, bid: float, ask: float, base_bal: float, quote_bal: float) -> None:
        # 1) Update fills first (so realized profit appears)
        self._sync_live_fills_from_balances(base_bal, quote_bal)

        # 2) Respect max_open_orders (your state is single open_order; keep it simple)
        if self.open_order:
            age = time.time() - float(self.open_order.get("ts", time.time()))
            if age > self.cfg.order_ttl_s:
                self._set_open_order(None)
                return

            cur_price = float(self.open_order.get("price", 0.0))
            side = self.open_order.get("side")
            target = (bid if side == "SELL" else ask)
            if cur_price > 0:
                move_bps = abs((target - cur_price) / cur_price) * 10000.0
                if move_bps < self.cfg.requote_bps:
                    return

            # moved enough → drop state and place a fresh offer next loop
            self._set_open_order(None)
            return

        # 3) If we have a position, try to exit for realized profit (even if spread isn't huge)
        force_exit, reason = self._should_force_exit(bid)
        if self.position and force_exit and self._cooldown_ok():
            base_qty = float(self.position.get("base_qty", 0.0))
            base_qty = max(0.0, min(base_qty, base_bal))
            if base_qty >= self.cfg.min_base_lot:
                sell_price = bid  # more likely to fill (realize profit)
                self._submit_sell_offer(base_qty, sell_price)
                self._set_last_trade_ts(time.time())
                self._set_open_order({"ts": time.time(), "side": "SELL", "price": sell_price, "qty": base_qty, "reason": reason})
            return

        # 4) Normal entry condition: only open a new position when spread is good
        if not self._cooldown_ok():
            return

        spread_bps = self._spread_bps(bid, ask)
        if spread_bps < self.cfg.min_spread_bps:
            return

        # BUY at ask (more fill likelihood), only if no position (or you can allow scaling in)
        if quote_bal >= self.cfg.trade_quote_amount and (not self.position):
            buy_price = ask
            base_qty = self.cfg.trade_quote_amount / max(buy_price, 1e-12)
            if base_qty < self.cfg.min_base_lot:
                return
            self._submit_buy_offer(base_qty, buy_price)
            self._set_last_trade_ts(time.time())
            self._set_open_order({"ts": time.time(), "side": "BUY", "price": buy_price, "qty": base_qty})

    def _trade_sim(self, bid: float, ask: float) -> None:
        # Exit logic first (to realize profit)
        if self.position:
            force_exit, _reason = self._should_force_exit(bid)
            if force_exit and self._cooldown_ok():
                base_qty = float(self.position.get("base_qty", 0.0))
                base_qty = min(base_qty, self.sim_base)
                if base_qty >= self.cfg.min_base_lot:
                    # sell at bid
                    quote_got = base_qty * bid
                    self.sim_base -= base_qty
                    self.sim_quote += quote_got
                    realized = self._apply_sell_from_position(now_utc(), base_qty, quote_got, bid, "SIM")
                    print(f"REALIZED (SIM) = {realized:.6f} {self.cfg.quote_asset}")
                    self._set_last_trade_ts(time.time())
                return

        # Normal entry condition
        if not self._cooldown_ok():
            return

        spread_bps = self._spread_bps(bid, ask)
        if spread_bps < self.cfg.min_spread_bps:
            return

        # Only buy if no position (so profit is easier to track/realize)
        if not self.position and self.sim_quote >= self.cfg.trade_quote_amount:
            base_qty = self.cfg.trade_quote_amount / ask  # conservative fill
            if base_qty < self.cfg.min_base_lot:
                return
            self.sim_quote -= self.cfg.trade_quote_amount
            self.sim_base += base_qty
            self._apply_buy_to_position(now_utc(), base_qty, self.cfg.trade_quote_amount, ask, "SIM")
            self._set_last_trade_ts(time.time())

    # -------------------------
    # Loop
    # -------------------------

    def run(self) -> None:
        print("Starting bot loop...")
        while True:
            t0 = time.time()
            try:
                bid, ask = fetch_best_bid_ask(self.server, self.base_asset, self.quote_asset)
                spread_bps = self._spread_bps(bid, ask)

                if self.cfg.is_live:
                    base_bal, quote_bal = fetch_balances(self.server, self.cfg)
                    self._trade_live(bid, ask, base_bal, quote_bal)
                    # refresh balances after potential fills/submit
                    base_bal, quote_bal = fetch_balances(self.server, self.cfg)
                else:
                    base_bal, quote_bal = self.sim_base, self.sim_quote
                    self._trade_sim(bid, ask)
                    base_bal, quote_bal = self.sim_base, self.sim_quote

                self._maybe_snapshot(bid, ask, spread_bps, base_bal, quote_bal)

                open_order_str = "none"
                if self.open_order:
                    side = self.open_order.get("side")
                    price = float(self.open_order.get("price"))
                    qty = float(self.open_order.get("qty"))
                    age = int(time.time() - float(self.open_order.get("ts", time.time())))
                    reason = self.open_order.get("reason")
                    if reason:
                        open_order_str = f"{side}@{price:.7f} qty={qty:.6f} age={age}s ({reason})"
                    else:
                        open_order_str = f"{side}@{price:.7f} qty={qty:.6f} age={age}s"

                print(
                    f"TICK bid={bid:.7f} ask={ask:.7f} "
                    f"spread={spread_bps/100:.4f}% "
                    f"bal{self.cfg.quote_asset}={quote_bal:.2f} "
                    f"bal{self.cfg.base_asset}={base_bal:.6f}"
                )

                # Print report roughly every snapshot interval boundary
                if int(time.time()) % max(self.cfg.snapshot_interval_s, 1) == 0:
                    self._report(bid, ask, base_bal, quote_bal, open_order_str)

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

