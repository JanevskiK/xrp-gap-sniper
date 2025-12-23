#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# =========================
# Helpers: env parsing
# =========================

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v

def env_float(name: str, default: float) -> float:
    v = _env(name, "")
    return default if v.strip() == "" else float(v)

def env_int(name: str, default: int) -> int:
    v = _env(name, "")
    return default if v.strip() == "" else int(float(v))

def env_bool(name: str, default: bool = False) -> bool:
    v = _env(name, "")
    if v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# =========================
# Config (matches your vars)
# =========================

HORIZON_URL = _env("HORIZON_URL", "https://horizon.stellar.org").strip().strip('"')

BASE_ASSET_TYPE = _env("BASE_ASSET_TYPE", "native").strip().strip('"')  # "native" only in your config
BASE_ASSET_CODE = _env("BASE_ASSET_CODE", "").strip().strip('"')
BASE_ASSET_ISSUER = _env("BASE_ASSET_ISSUER", "").strip().strip('"')

COUNTER_ASSET_TYPE = _env("COUNTER_ASSET_TYPE", "credit_alphanum4").strip().strip('"')
COUNTER_ASSET_CODE = _env("COUNTER_ASSET_CODE", "USDC").strip().strip('"')
COUNTER_ASSET_ISSUER = _env("COUNTER_ASSET_ISSUER", "").strip().strip('"')

POLL_SECONDS = env_float("POLL_SECONDS", 1.0)
TRADE_COUNTER_AMOUNT = env_float("TRADE_COUNTER_AMOUNT", 100.0)

MIN_SPREAD_PCT = env_float("MIN_SPREAD_PCT", 0.03)      # percent
MIN_DEPTH_MULT = env_float("MIN_DEPTH_MULT", 0.02)      # fraction of TRADE_COUNTER_AMOUNT needed at top of book

PRINT_EVERY = env_int("PRINT_EVERY", 1)
REPORT_EVERY_SECONDS = env_int("REPORT_EVERY_SECONDS", 60)
ROLLING_HOURS = env_int("ROLLING_HOURS", 48)

INITIAL_USDC = env_float("INITIAL_USDC", 1000.0)
INITIAL_XLM = env_float("INITIAL_XLM", 0.0)

REQUOTE_SECONDS = env_int("REQUOTE_SECONDS", 60)
REQUOTE_BPS = env_int("REQUOTE_BPS", 10)  # used as informational here

ORDER_TIMEOUT_SECONDS = env_int("ORDER_TIMEOUT_SECONDS", 900)
QUEUE_JOIN_FACTOR = env_float("QUEUE_JOIN_FACTOR", 1.0)
FILL_RATE_CAP = env_float("FILL_RATE_CAP", 999999.0)  # not limiting in this paper version

NETWORK_FEE_XLM = env_float("NETWORK_FEE_XLM", 0.00001)
OPS_PER_FILL = env_int("OPS_PER_FILL", 2)

MAX_INVENTORY_XLM = env_float("MAX_INVENTORY_XLM", 2000.0)
MIN_INVENTORY_XLM = env_float("MIN_INVENTORY_XLM", 0.0)

TARGET_XLM = env_float("TARGET_XLM", 500.0)
BAND_XLM = env_float("BAND_XLM", 150.0)

STATS_FILE = _env("STATS_FILE", "/tmp/stellar_mm_trades.jsonl").strip().strip('"')

PAPER_TRADING = env_bool("PAPER_TRADING", True)  # should be True for you


# =========================
# Data models
# =========================

@dataclass
class Trade:
    ts: float
    action: str               # "BUY" or "SELL"
    price: float              # USDC per XLM
    xlm: float
    usdc: float               # signed delta: BUY negative, SELL positive
    spread_pct: float
    note: str = ""
    fee_xlm: float = 0.0

@dataclass
class State:
    started_ts: float
    balUSDC: float
    balXLM: float
    trades: List[Trade]
    last_requote_ts: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_ts": self.started_ts,
            "balUSDC": self.balUSDC,
            "balXLM": self.balXLM,
            "last_requote_ts": self.last_requote_ts,
            "trades": [asdict(t) for t in self.trades],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "State":
        trades = [Trade(**t) for t in d.get("trades", [])]
        return State(
            started_ts=float(d.get("started_ts", time.time())),
            balUSDC=float(d.get("balUSDC", INITIAL_USDC)),
            balXLM=float(d.get("balXLM", INITIAL_XLM)),
            last_requote_ts=float(d.get("last_requote_ts", 0.0)),
            trades=trades,
        )


# =========================
# Persistence
# =========================

def load_state() -> State:
    if not os.path.exists(STATS_FILE):
        return State(time.time(), INITIAL_USDC, INITIAL_XLM, [], 0.0)
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return State(time.time(), INITIAL_USDC, INITIAL_XLM, [], 0.0)
            return State.from_dict(json.loads(txt))
    except Exception:
        # don't crash due to a bad file
        return State(time.time(), INITIAL_USDC, INITIAL_XLM, [], 0.0)

def save_state(state: State) -> None:
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


# =========================
# Horizon orderbook
# =========================

def _asset_params(prefix: str, asset_type: str, code: str, issuer: str) -> Dict[str, str]:
    if asset_type == "native":
        return {f"{prefix}_asset_type": "native"}
    if issuer.strip() == "" or code.strip() == "":
        raise RuntimeError(f"Missing {prefix} asset code/issuer for non-native asset.")
    # Horizon accepts credit_alphanum4 or credit_alphanum12
    return {
        f"{prefix}_asset_type": asset_type,
        f"{prefix}_asset_code": code,
        f"{prefix}_asset_issuer": issuer,
    }

async def fetch_orderbook(session: aiohttp.ClientSession) -> Dict[str, Any]:
    url = f"{HORIZON_URL.rstrip('/')}/order_book"
    params: Dict[str, str] = {}
    # selling = BASE, buying = COUNTER (this yields prices in COUNTER per BASE)
    params.update(_asset_params("selling", BASE_ASSET_TYPE, BASE_ASSET_CODE, BASE_ASSET_ISSUER))
    params.update(_asset_params("buying", COUNTER_ASSET_TYPE, COUNTER_ASSET_CODE, COUNTER_ASSET_ISSUER))

    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        return await r.json()

def _best_levels(ob: Dict[str, Any]) -> Tuple[float, float, float, float]:
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        raise RuntimeError("Orderbook empty (no bids or asks).")
    bid_price = float(bids[0]["price"])
    bid_xlm = float(bids[0]["amount"])
    ask_price = float(asks[0]["price"])
    ask_xlm = float(asks[0]["amount"])
    return bid_price, ask_price, bid_xlm, ask_xlm

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 0.0
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    return (ask - bid) / mid * 100.0


# =========================
# Profit window
# =========================

def rolling_profit(state: State, hours: int) -> Tuple[float, int]:
    cutoff = time.time() - hours * 3600
    prof = 0.0
    n = 0
    for t in state.trades:
        if t.ts >= cutoff:
            prof += t.usdc
            n += 1
    return prof, n

def prune_old(state: State, keep_hours: int = 24 * 14) -> None:
    cutoff = time.time() - keep_hours * 3600
    state.trades = [t for t in state.trades if t.ts >= cutoff]


# =========================
# Paper execution model
# =========================

def within_band(state: State) -> bool:
    # if inventory is far from target, you may want to bias behavior
    return (TARGET_XLM - BAND_XLM) <= state.balXLM <= (TARGET_XLM + BAND_XLM)

def apply_fee(state: State, fee_xlm: float) -> None:
    if fee_xlm <= 0:
        return
    state.balXLM = max(0.0, state.balXLM - fee_xlm)

async def try_trade_paper(
    state: State,
    bid: float,
    ask: float,
    bid_xlm: float,
    ask_xlm: float,
    spr: float,
) -> Optional[str]:
    """
    Simulate:
      - if spread >= MIN_SPREAD_PCT and top-of-book depth sufficient:
        BUY at ask using TRADE_COUNTER_AMOUNT USDC
        wait a short moment based on QUEUE_JOIN_FACTOR and POLL_SECONDS
        SELL at bid
    """
    if not PAPER_TRADING:
        raise RuntimeError("This main.py is paper-only. Set PAPER_TRADING=1.")

    if spr < MIN_SPREAD_PCT:
        return None

    # Depth check: require a fraction of trade size to exist at top-of-book
    # Convert top XLM depth to USDC depth using price.
    bid_depth_usdc = bid_xlm * bid
    ask_depth_usdc = ask_xlm * ask
    min_needed = TRADE_COUNTER_AMOUNT * MIN_DEPTH_MULT

    if bid_depth_usdc < min_needed or ask_depth_usdc < min_needed:
        return "skip(depth)"

    # Choose trade size limited by balances & inventory caps
    usdc_spend = min(TRADE_COUNTER_AMOUNT, state.balUSDC)
    if usdc_spend <= 0:
        return "skip(no-usdc)"

    xlm_buy = usdc_spend / ask
    xlm_buy = min(xlm_buy, ask_xlm)  # don't exceed top ask amount (simple)

    if state.balXLM + xlm_buy > MAX_INVENTORY_XLM:
        return "skip(max-inv)"

    # Simulated "queue join" delay (very simplified)
    queue_delay = max(0.0, POLL_SECONDS * QUEUE_JOIN_FACTOR)
    await asyncio.sleep(queue_delay)

    # Simulated fees
    total_fee_xlm = NETWORK_FEE_XLM * OPS_PER_FILL

    # BUY
    state.balUSDC -= usdc_spend
    state.balXLM += xlm_buy
    apply_fee(state, total_fee_xlm / 2)

    state.trades.append(Trade(
        ts=time.time(),
        action="BUY",
        price=ask,
        xlm=xlm_buy,
        usdc=-usdc_spend,
        spread_pct=spr,
        note="paper",
        fee_xlm=total_fee_xlm / 2,
    ))

    # Sell only if above min inventory
    xlm_sell = xlm_buy
    if state.balXLM - xlm_sell < MIN_INVENTORY_XLM:
        return "bought-but-sell-blocked(min-inv)"

    # Another small delay to mimic fill / timeout behavior
    await asyncio.sleep(min(2.0, max(0.0, POLL_SECONDS)))

    # SELL at bid, limited by top bid liquidity
    xlm_sell = min(xlm_sell, bid_xlm)
    if xlm_sell <= 0:
        return "bought-but-no-bid"

    usdc_get = xlm_sell * bid

    state.balXLM -= xlm_sell
    state.balUSDC += usdc_get
    apply_fee(state, total_fee_xlm / 2)

    state.trades.append(Trade(
        ts=time.time(),
        action="SELL",
        price=bid,
        xlm=xlm_sell,
        usdc=usdc_get,
        spread_pct=spr,
        note="paper",
        fee_xlm=total_fee_xlm / 2,
    ))

    pnl = (-usdc_spend + usdc_get)
    return f"TRADE pnl={pnl:+.6f} {COUNTER_ASSET_CODE}"

def fmt_tick(bid: float, ask: float, bid_xlm: float, ask_xlm: float, state: State) -> str:
    spr = spread_pct(bid, ask)
    return (
        f"TICK bid={bid:.7f} ask={ask:.7f} spread={spr:.4f}% "
        f"bid_xlm={bid_xlm:.2f} ask_xlm={ask_xlm:.2f} "
        f"balUSDC={state.balUSDC:.2f} balXLM={state.balXLM:.6f}"
    )

def fmt_report(state: State) -> str:
    prof, n = rolling_profit(state, ROLLING_HOURS)
    return f"REPORT {ROLLING_HOURS}h trades={n} profit={prof:.6f} {COUNTER_ASSET_CODE}"


# =========================
# Main loop
# =========================

async def main_loop() -> None:
    # sanity checks
    if BASE_ASSET_TYPE != "native":
        raise RuntimeError("This config expects BASE_ASSET_TYPE=native for XLM base.")
    if COUNTER_ASSET_TYPE not in ("credit_alphanum4", "credit_alphanum12"):
        raise RuntimeError("COUNTER_ASSET_TYPE must be credit_alphanum4 or credit_alphanum12.")
    if COUNTER_ASSET_ISSUER.strip() == "":
        raise RuntimeError("COUNTER_ASSET_ISSUER is required for USDC-like assets.")

    state = load_state()
    last_print = 0.0
    last_report = 0.0

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                ob = await fetch_orderbook(session)
                bid, ask, bid_xlm, ask_xlm = _best_levels(ob)
                spr = spread_pct(bid, ask)

                now = time.time()

                if now - last_print >= max(1, PRINT_EVERY):
                    print(fmt_tick(bid, ask, bid_xlm, ask_xlm, state), flush=True)
                    last_print = now

                action = await try_trade_paper(state, bid, ask, bid_xlm, ask_xlm, spr)
                if action:
                    print(action, flush=True)

                prune_old(state)
                save_state(state)

                if now - last_report >= max(1, REPORT_EVERY_SECONDS):
                    print(fmt_report(state), flush=True)
                    last_report = now

                await asyncio.sleep(max(0.1, POLL_SECONDS))

            except aiohttp.ClientResponseError as e:
                print(f"HTTP error: {e.status} {e.message}", flush=True)
                await asyncio.sleep(2)
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(2)


def main() -> None:
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()

