#!/usr/bin/env python3
"""
Stellar XLM/USDC market-making bot (simple spread capture).

Reads config from environment variables (prints ENV CHECK + resolved config),
tracks trades/snapshots in SQLite, and can run in:
- LIVE mode: if SECRET_KEY is provided (submits real orders)
- SIM mode : if SECRET_KEY is missing (no submissions; logs signals only)

Required for USDC (or any credit asset):
- QUOTE_ISSUER

Recommended env vars (yours):
BASE_ASSET="XLM"
QUOTE_ASSET="USDC"
MIN_SPREAD_BPS="35"
TRADE_COOLDOWN_S="600"
TRADE_QUOTE_AMOUNT="25"
MIN_BASE_LOT="1.0"
START_PRICE="0.215"
SIM_VOL_BPS="8"
SIM_SPREAD_BPS="45"
TICK_INTERVAL_S="2"
SNAPSHOT_INTERVAL_S="30"
REQUOTE_BPS="10"
STARTING_QUOTE="1000"
STARTING_BASE="500"
DB_PATH="/tmp/bot.db"

Extra supported:
HORIZON_URL="https://horizon.stellar.org"
SECRET_KEY="S...."  (enables LIVE)
BASE_ISSUER="..."   (if BASE_ASSET is non-native)
QUOTE_ISSUER="..."  (required for USDC)
MAX_OPEN_ORDERS="2"
ORDER_TTL_S="900"
"""

from __future__ import annotations

import os
import time
import math
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List

# stellar-sdk
from stellar_sdk import Server, Keypair, TransactionBuilder, Network, Asset
from stellar_sdk.exceptions import BadRequestError, NotFoundError


# -------------------------
# Helpers / config parsing
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def bps_to_pct(bps: float) -> float:
    return bps / 10000.0

def pct_to_bps(pct: float) -> float:
    return pct * 10000.0

def safe_float(s: Any, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


@dataclass
class Config:
    # trading pair
    base_asset: str
    quote_asset: str
    base_issuer: Optional[str]
    quote_issuer: Optional[str]

    # behavior
    min_spread_bps: float
    requote_bps: float
    trade_cooldown_s: int
    trade_quote_amount: float
    min_base_lot: float

    # timing
    tick_interval_s: int
    snapshot_interval_s: int

    # simulation knobs (used only for SIM fills / synthetic movement if you expand)
    start_price: float
    sim_vol_bps: float
    sim_spread_bps: float

    # starting balances (SIM reference; LIVE reads actual balances)
    starting_quote: float
    starting_base: float

    # infra
    horizon_url: str
    secret_key: Optional[str]
    db_path: str

    # extras
    max_open_orders: int
    order_ttl_s: int

    @property
    def is_live(self) -> bool:
        return bool(self.secret_key and self.secret_key.startswith("S"))

    def validate(self) -> None:
        if self.quote_asset.lower() != "native" and self.quote_asset.upper() != "XLM":
            # credit asset
            if not self.quote_issuer:
                raise ValueError("QUOTE_ISSUER is required for non-native QUOTE_ASSET (e.g., USDC).")
        if self.base_asset.lower() != "native" and self.base_asset.upper() != "XLM":
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


def load_config() -> Config:
    # Your keys
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
    )
    cfg.validate()
    return cfg


def print_env_check() -> None:
    keys = [
        "BASE_ASSET","QUOTE_ASSET","BASE_ISSUER","QUOTE_ISSUER",
        "MIN_SPREAD_BPS","TRADE_COOLDOWN_S","TRADE_QUOTE_AMOUNT",
        "MIN_BASE_LOT","START_PRICE","SIM_VOL_BPS","SIM_SPREAD_BPS",
        "TICK_INTERVAL_S","SNAPSHOT_INTERVAL_S","REQUOTE_BPS",
        "STARTING_QUOTE","STARTING_BASE","DB_PATH","HORIZON_URL",
        "SECRET_KEY","MAX_OPEN_ORDERS","ORDER_TTL_S"
    ]
    print("=== ENV CHECK ===")
    for k in keys:
        v = os.getenv(k)
        if k == "SECRET_KEY" and v:
            v = v[:5] + "..." + v[-5:]
        print(f"{k}={v!r}")
    print("=================")


# -------------------------
# DB layer
# -------------------------

class DB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            side TEXT NOT NULL,           -- BUY or SELL (base side)
            price REAL NOT NULL,          -- quote per base
            base_qty REAL NOT NULL,
            quote_qty REAL NOT NULL,
            realized_pnl_quote REAL NOT NULL DEFAULT 0.0,
            mode TEXT NOT NULL            -- SIM or LIVE
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
        cur.execute("INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                    (k, json.dumps(v)))
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
            INSERT INTO trades(ts, side, price, base_qty, quote_qty, realized_pnl_quote, mode)
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
            kwargs["bid"],
            kwargs["ask"],
            kwargs["spread_bps"],
            kwargs["base_bal"],
            kwargs["quote_bal"],
            kwargs["equity_quote"],
            kwargs["unrealized_quote"],
            kwargs["realized_total_quote"],
            kwargs.get("open_order_json"),
        ))
        self.conn.commit()


# -------------------------
# Stellar market / balances
# -------------------------

def make_asset(code: str, issuer: Optional[str]) -> Asset:
    if code.upper() in ("XLM", "NATIVE"):
        return Asset.native()
    if not issuer:
        raise ValueError(f"Issuer required for asset {code}")
    return Asset(code.upper(), issuer)

def fetch_best_bid_ask(server: Server, selling: Asset, buying: Asset) -> Tuple[float, float]:
    """
    For orderbook endpoint:
      - "selling" is base asset you give
      - "buying"  is asset you receive
    We want price as QUOTE per BASE.
    So we query orderbook for:
      selling=BASE, buying=QUOTE
    In response:
      bids: people want to BUY selling (BASE) and pay buying (QUOTE) => bid price (QUOTE/BASE)
      asks: people want to SELL selling (BASE) for buying (QUOTE) => ask price (QUOTE/BASE)
    """
    ob = server.orderbook(selling=selling, buying=buying).call()
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
    """
    Returns (base_balance, quote_balance) for the account in LIVE.
    """
    kp = Keypair.from_secret(cfg.secret_key)  # type: ignore[arg-type]
    acc = server.accounts().account_id(kp.public_key).call()
    base = cfg.base_asset
    quote = cfg.quote_asset

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

    base_bal = bal_for(base, cfg.base_issuer)
    quote_bal = bal_for(quote, cfg.quote_issuer)
    return base_bal, quote_bal


# -------------------------
# Trading logic
# -------------------------

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(cfg.db_path)
        self.server = Server(horizon_url=cfg.horizon_url)

        self.base_asset = make_asset(cfg.base_asset, cfg.base_issuer)
        self.quote_asset = make_asset(cfg.quote_asset, cfg.quote_issuer)

        # state
        self.last_trade_ts = self._load_last_trade_ts()
        self.open_order = self.db.get_state("open_order", default=None)  # dict or None
        self.last_snapshot_ts = 0.0

        # SIM balances (only used if not live)
        self.sim_base = float(cfg.starting_base)
        self.sim_quote = float(cfg.starting_quote)

        print("=== CONFIG RESOLVED ===")
        d = asdict(cfg)
        if d.get("secret_key"):
            d["secret_key"] = (d["secret_key"][:5] + "..." + d["secret_key"][-5:])
        print(json.dumps(d, indent=2))
        print("=======================")
        print(f"MODE: {'LIVE' if cfg.is_live else 'SIM'}")

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

    def _cooldown_ok(self) -> bool:
        return (time.time() - self.last_trade_ts) >= self.cfg.trade_cooldown_s

    def _spread_bps(self, bid: float, ask: float) -> float:
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid) * 10000.0

    def _equity_quote(self, bid: float, base_bal: float, quote_bal: float) -> float:
        # value base at bid into quote
        return quote_bal + base_bal * bid

    def _unrealized_quote(self, bid: float, equity: float) -> float:
        # relative to starting total
        start_equity = self.cfg.starting_quote + self.cfg.starting_base * self.cfg.start_price
        return equity - start_equity

    def _report(self, bid: float, ask: float, base_bal: float, quote_bal: float, open_order_str: str) -> None:
        since = now_utc() - timedelta(hours=48)
        trades_48h = self.db.count_trades_since(since)
        realized_48h = self.db.realized_pnl_since(since)
        realized_total = self.db.realized_pnl_total()

        equity = self._equity_quote(bid, base_bal, quote_bal)
        unreal = self._unrealized_quote(bid, equity)
        total_now = realized_total + unreal  # simple combined view

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
        unreal = self._unrealized_quote(bid, equity)
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

    def _place_or_requote_orders_live(self, bid: float, ask: float) -> None:
        """
        Simple strategy:
        - When spread >= MIN_SPREAD_BPS and cooldown ok:
            Place BOTH:
              BUY base with quote slightly below mid (near bid)
              SELL base for quote slightly above mid (near ask)
        - Keep at most max_open_orders (we manage via single "open_order" record)
        - Requote if market moved more than REQUOTE_BPS from our stored price.
        """
        # For simplicity: we manage ONE active "pair" conceptually; real management depends on offers list.
        # This bot uses "manage_sell_offer" only and doesn't attempt sophisticated inventory mgmt.
        # It is intentionally minimal + debuggable.

        if not self._cooldown_ok():
            return

        spread = self._spread_bps(bid, ask)
        if spread < self.cfg.min_spread_bps:
            return

        # Determine target prices
        mid = (bid + ask) / 2.0
        buy_price = bid  # conservative
        sell_price = ask

        # If we already have an open order record, requote if needed
        if self.open_order:
            # If it's too old, we drop state (actual offer may still exist; you can extend to cancel)
            age = time.time() - float(self.open_order.get("ts", time.time()))
            if age > self.cfg.order_ttl_s:
                self._set_open_order(None)
                return

            cur_price = float(self.open_order.get("price", 0.0))
            side = self.open_order.get("side")
            # requote threshold relative to current market
            target = sell_price if side == "SELL" else buy_price
            if cur_price > 0:
                move_bps = abs((target - cur_price) / cur_price) * 10000.0
                if move_bps < self.cfg.requote_bps:
                    return
            # If moved enough: clear local state; next tick will place a fresh one
            self._set_open_order(None)
            return

        # Decide what to place now: we place a SELL (inventory-based) if we have base; otherwise BUY.
        base_bal, quote_bal = fetch_balances(self.server, self.cfg)

        # Convert quote amount into base qty at chosen price
        base_qty_for_quote = self.cfg.trade_quote_amount / max(buy_price, 1e-12)

        # Inventory choice:
        if base_bal >= max(self.cfg.min_base_lot, base_qty_for_quote * 0.9):
            # Place SELL base for quote
            base_qty = max(self.cfg.min_base_lot, min(base_bal, base_qty_for_quote))
            self._submit_sell_offer(base_qty=base_qty, price=sell_price)
            self._set_last_trade_ts(time.time())
            self._set_open_order({
                "ts": time.time(),
                "side": "SELL",
                "price": sell_price,
                "qty": base_qty
            })
        elif quote_bal >= self.cfg.trade_quote_amount:
            # Place BUY base with quote by selling quote asset? On Stellar, easiest is manage_buy_offer,
            # but SDK supports it. We'll do manage_buy_offer.
            base_qty = max(self.cfg.min_base_lot, base_qty_for_quote)
            self._submit_buy_offer(base_qty=base_qty, price=buy_price)
            self._set_last_trade_ts(time.time())
            self._set_open_order({
                "ts": time.time(),
                "side": "BUY",
                "price": buy_price,
                "qty": base_qty
            })
        else:
            # Not enough inventory
            return

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
            print("LIVE submit SELL offer failed:", str(e)[:300])
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
                selling=self.quote_asset,   # we pay quote
                buying=self.base_asset,     # we receive base
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
            print("LIVE submit BUY offer failed:", str(e)[:300])
            raise

    def _sim_step(self, bid: float, ask: float) -> None:
        """
        SIM logic: if spread >= threshold & cooldown ok, we "pretend fill" by crossing:
        - If we have quote: BUY at ask (worse case)
        - If we have base : SELL at bid
        This is intentionally conservative (helps you debug conditions).
        """
        if not self._cooldown_ok():
            return

        spread = self._spread_bps(bid, ask)
        if spread < self.cfg.min_spread_bps:
            return

        # Choose action based on inventory
        if self.sim_quote >= self.cfg.trade_quote_amount:
            # BUY base at ask
            base_qty = self.cfg.trade_quote_amount / ask
            if base_qty < self.cfg.min_base_lot:
                return
            self.sim_quote -= self.cfg.trade_quote_amount
            self.sim_base += base_qty
            self.db.add_trade(
                ts=now_utc(),
                side="BUY",
                price=ask,
                base_qty=base_qty,
                quote_qty=self.cfg.trade_quote_amount,
                realized_pnl_quote=0.0,
                mode="SIM"
            )
            self._set_last_trade_ts(time.time())
        elif self.sim_base >= self.cfg.min_base_lot:
            # SELL base at bid for approximately trade_quote_amount worth (or as much as possible)
            base_qty_target = self.cfg.trade_quote_amount / bid
            base_qty = min(self.sim_base, max(self.cfg.min_base_lot, base_qty_target))
            quote_got = base_qty * bid
            self.sim_base -= base_qty
            self.sim_quote += quote_got
            self.db.add_trade(
                ts=now_utc(),
                side="SELL",
                price=bid,
                base_qty=base_qty,
                quote_qty=quote_got,
                realized_pnl_quote=0.0,
                mode="SIM"
            )
            self._set_last_trade_ts(time.time())

    def run(self) -> None:
        print("Starting bot loop...")
        while True:
            t0 = time.time()
            try:
                bid, ask = fetch_best_bid_ask(self.server, selling=self.base_asset, buying=self.quote_asset)
                spread_bps = self._spread_bps(bid, ask)

                if self.cfg.is_live:
                    base_bal, quote_bal = fetch_balances(self.server, self.cfg)
                else:
                    base_bal, quote_bal = self.sim_base, self.sim_quote

                # trading step
                if self.cfg.is_live:
                    self._place_or_requote_orders_live(bid, ask)
                else:
                    self._sim_step(bid, ask)

                # Snapshot + report
                self._maybe_snapshot(bid, ask, spread_bps, base_bal, quote_bal)

                open_order_str = "none"
                if self.open_order:
                    side = self.open_order.get("side")
                    price = self.open_order.get("price")
                    qty = self.open_order.get("qty")
                    age = int(time.time() - float(self.open_order.get("ts", time.time())))
                    open_order_str = f"{side}@{price:.7f} qty={qty:.6f} age={age}s"

                # Print a compact tick line
                print(
                    f"TICK bid={bid:.7f} ask={ask:.7f} "
                    f"spread={spread_bps:.4f}% "
                    f"bal{self.cfg.quote_asset}={quote_bal:.2f} "
                    f"bal{self.cfg.base_asset}={base_bal:.6f} "
                )

                # Print report every snapshot interval boundary (or you can adjust)
                if int(time.time()) % max(self.cfg.snapshot_interval_s, 1) == 0:
                    self._report(bid, ask, base_bal, quote_bal, open_order_str)

            except Exception as e:
                print("ERROR:", str(e))

            # sleep
            elapsed = time.time() - t0
            sleep_s = max(0.0, self.cfg.tick_interval_s - elapsed)
            time.sleep(sleep_s)


def main():
    print_env_check()
    cfg = load_config()
    bot = Bot(cfg)
    bot.run()


if __name__ == "__main__":
    main()

