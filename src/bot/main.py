#!/usr/bin/env python3
"""
Paper-trading bot for Stellar orderbook spreads (e.g., XLM/USDC).
- NO real trades are submitted.
- NO Stellar secret seed is required.
- Reads best bid/ask from Horizon orderbook (public GET endpoint).
- Simulates: if spread >= threshold, buy at ask then sell at bid after HOLD_SECONDS.
- Persists balances + trades to STATS_FILE (JSON).
- Reports rolling profit over ROLLING_HOURS (default 48).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# -----------------------------
# Config
# -----------------------------

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return float(v)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return int(v)

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v

HORIZON_URL = env_str("HORIZON_URL", "https://horizon.stellar.org")

# Assets for the orderbook:
# base = what you're pricing (XLM by default)
# counter = what you pay/receive (USDC by default)
BASE_CODE = env_str("BASE_CODE", "XLM")
BASE_ISSUER = env_str("BASE_ISSUER", "")  # empty means native XLM

COUNTER_CODE = env_str("COUNTER_CODE", "USDC")
COUNTER_ISSUER = env_str("COUNTER_ISSUER", "")  # REQUIRED for non-native assets (like USDC)

# Trading behavior
REPORT_EVERY_SECONDS = env_int("REPORT_EVERY_SECONDS", 60)
HOLD_SECONDS = env_int("HOLD_SECONDS", 5)  # how long to wait between buy & sell simulation
SPREAD_TRIGGER_PCT = env_float("SPREAD_TRIGGER_PCT", 0.05)  # e.g. 0.05 = 0.05%
TRADE_USDC = env_float("TRADE_USDC", 10.0)  # how much counter asset to spend per simulated buy (in USDC)
NETWORK_FEE_XLM = env_float("NETWORK_FEE_XLM", 0.00001)  # simulated fee per operation (optional)
OPS_PER_FILL = env_int("OPS_PER_FILL", 2)  # buy + sell

# Inventory / risk controls (paper only)
MIN_INVENTORY_XLM = env_float("MIN_INVENTORY_XLM", 0.0)
MAX_INVENTORY_XLM = env_float("MAX_INVENTORY_XLM", 2000.0)

# Stats persistence
STATS_FILE = env_str("STATS_FILE", "/tmp/stellar_mm_trades.jsonl")  # jsonl-ish; we store one JSON object file
ROLLING_HOURS = env_int("ROLLING_HOURS", 48)

# If you want deterministic starting balances, set these:
START_USDC = env_float("START_USDC", 1000.0)
START_XLM = env_float("START_XLM", 0.0)

# Safety: force paper mode always (no secrets needed)
PAPER_TRADING = True


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Trade:
    ts: float
    side: str  # "BUY" or "SELL"
    price: float  # counter per base (USDC per XLM)
    base_amount: float
    counter_amount: float  # USDC delta (negative for buy, positive for sell)
    fee_xlm: float = 0.0
    note: str = ""

@dataclass
class State:
    bal_usdc: float
    bal_xlm: float
    trades: List[Trade]
    started_ts: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "bal_usdc": self.bal_usdc,
            "bal_xlm": self.bal_xlm,
            "started_ts": self.started_ts,
            "trades": [asdict(t) for t in self.trades],
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "State":
        trades = [Trade(**t) for t in d.get("trades", [])]
        return State(
            bal_usdc=float(d.get("bal_usdc", START_USDC)),
            bal_xlm=float(d.get("bal_xlm", START_XLM)),
            started_ts=float(d.get("started_ts", time.time())),
            trades=trades,
        )


# -----------------------------
# Persistence
# -----------------------------

def load_state() -> State:
    if not os.path.exists(STATS_FILE):
        return State(bal_usdc=START_USDC, bal_xlm=START_XLM, trades=[], started_ts=time.time())
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return State(bal_usdc=START_USDC, bal_xlm=START_XLM, trades=[], started_ts=time.time())
            d = json.loads(raw)
            return State.from_json(d)
    except Exception:
        # If file corrupted, start fresh but don't crash
        return State(bal_usdc=START_USDC, bal_xlm=START_XLM, trades=[], started_ts=time.time())

def save_state(state: State) -> None:
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(state.to_json(), f, ensure_ascii=False, indent=2)


# -----------------------------
# Horizon orderbook fetch
# -----------------------------

def asset_query_params(prefix: str, code: str, issuer: str) -> Dict[str, str]:
    """
    Horizon expects:
      - native asset: <prefix>_asset_type=native
      - non-native: <prefix>_asset_type=credit_alphanum4|credit_alphanum12 + code + issuer
    """
    if issuer == "" and code.upper() == "XLM":
        return {f"{prefix}_asset_type": "native"}

    if issuer == "":
        raise RuntimeError(
            f"Missing {prefix.upper()} issuer for non-native asset {code}. "
            f"Set {prefix.upper()}_ISSUER env var."
        )

    asset_type = "credit_alphanum4" if len(code) <= 4 else "credit_alphanum12"
    return {
        f"{prefix}_asset_type": asset_type,
        f"{prefix}_asset_code": code,
        f"{prefix}_asset_issuer": issuer,
    }

async def fetch_best_bid_ask(session: aiohttp.ClientSession) -> Tuple[float, float, float, float]:
    """
    Returns: (best_bid_price, best_ask_price, best_bid_amount_base, best_ask_amount_base)
    Price is COUNTER per BASE.
    """
    params = {}
    params.update(asset_query_params("selling", BASE_CODE, BASE_ISSUER))
    params.update(asset_query_params("buying", COUNTER_CODE, COUNTER_ISSUER))

    url = f"{HORIZON_URL.rstrip('/')}/order_book"
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        data = await r.json()

    bids = data.get("bids", [])
    asks = data.get("asks", [])
    if not bids or not asks:
        raise RuntimeError("Orderbook empty (no bids or asks).")

    best_bid = float(bids[0]["price"])         # counter/base
    best_bid_amt = float(bids[0]["amount"])    # base amount available at bid
    best_ask = float(asks[0]["price"])         # counter/base
    best_ask_amt = float(asks[0]["amount"])    # base amount available at ask

    return best_bid, best_ask, best_bid_amt, best_ask_amt


# -----------------------------
# Paper trading logic
# -----------------------------

def pct_spread(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0 or ask < bid:
        return 0.0
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 100.0

def now_ts() -> float:
    return time.time()

def rolling_profit_usdc(state: State, hours: int) -> float:
    """
    Profit over last N hours based on net USDC change from SELL/BUY trade deltas.
    We treat:
      BUY  => counter_amount negative (spend USDC)
      SELL => counter_amount positive (receive USDC)
    Profit window = sum(counter_amount) over trades within window.
    """
    cutoff = now_ts() - hours * 3600
    p = 0.0
    for t in state.trades:
        if t.ts >= cutoff:
            p += float(t.counter_amount)
    return p

def prune_old_trades(state: State, keep_hours: int = 24 * 14) -> None:
    """
    Keep last ~2 weeks by default to avoid huge state files.
    Rolling profit uses ROLLING_HOURS anyway.
    """
    cutoff = now_ts() - keep_hours * 3600
    state.trades = [t for t in state.trades if t.ts >= cutoff]

def can_buy(state: State, usdc_to_spend: float, expected_base: float) -> bool:
    if state.bal_usdc < usdc_to_spend:
        return False
    # Ensure inventory won't exceed cap after buy
    if state.bal_xlm + expected_base > MAX_INVENTORY_XLM:
        return False
    return True

def can_sell(state: State, base_to_sell: float) -> bool:
    if state.bal_xlm - base_to_sell < MIN_INVENTORY_XLM:
        return False
    return True

def apply_fee_xlm(state: State, fee_xlm: float) -> None:
    # Fee taken from XLM balance for simulation. If you have no XLM, keep it 0.
    if fee_xlm <= 0:
        return
    state.bal_xlm = max(0.0, state.bal_xlm - fee_xlm)

def record_trade(state: State, trade: Trade) -> None:
    state.trades.append(trade)

async def simulate_cycle(
    state: State,
    bid: float,
    ask: float,
    bid_amt: float,
    ask_amt: float,
    spread_pct: float,
) -> Optional[str]:
    """
    If spread meets trigger, do paper buy and paper sell after HOLD_SECONDS.
    Returns a short note about action taken (or None if no trade).
    """
    if spread_pct < SPREAD_TRIGGER_PCT:
        return None

    # Buy with TRADE_USDC at ASK
    usdc_spend = min(TRADE_USDC, state.bal_usdc)
    if usdc_spend <= 0:
        return None

    base_buy = usdc_spend / ask  # XLM amount bought
    # Respect top-of-book available liquidity (paper approximation)
    base_buy = min(base_buy, ask_amt)

    if base_buy <= 0:
        return None

    if not can_buy(state, usdc_spend, base_buy):
        return "skip (buy blocked by balance/cap)"

    fee_total_xlm = NETWORK_FEE_XLM * OPS_PER_FILL

    # Apply BUY
    state.bal_usdc -= usdc_spend
    state.bal_xlm += base_buy
    apply_fee_xlm(state, fee_total_xlm / 2.0)

    record_trade(
        state,
        Trade(
            ts=now_ts(),
            side="BUY",
            price=ask,
            base_amount=base_buy,
            counter_amount=-usdc_spend,
            fee_xlm=fee_total_xlm / 2.0,
            note=f"spread={spread_pct:.4f}%",
        ),
    )

    # Wait then SELL at BID
    await asyncio.sleep(max(0, HOLD_SECONDS))

    base_sell = base_buy
    # Respect top-of-book bid liquidity too
    base_sell = min(base_sell, bid_amt)

    if base_sell <= 0:
        return "bought but cannot sell (no bid liquidity)"

    if not can_sell(state, base_sell):
        return "bought but sell blocked (min inventory)"

    usdc_receive = base_sell * bid

    state.bal_xlm -= base_sell
    state.bal_usdc += usdc_receive
    apply_fee_xlm(state, fee_total_xlm / 2.0)

    record_trade(
        state,
        Trade(
            ts=now_ts(),
            side="SELL",
            price=bid,
            base_amount=base_sell,
            counter_amount=usdc_receive,
            fee_xlm=fee_total_xlm / 2.0,
            note=f"spread={spread_pct:.4f}%",
        ),
    )

    return f"TRADE buy@{ask:.7f} sell@{bid:.7f} spread={spread_pct:.4f}% usdcΔ={(-usdc_spend + usdc_receive):+.6f}"


# -----------------------------
# Reporting
# -----------------------------

def tick_line(bid: float, ask: float, bid_amt: float, ask_amt: float, state: State) -> str:
    spread = pct_spread(bid, ask)
    return (
        f"TICK bid={bid:.7f} ask={ask:.7f} "
        f"spread={spread:.4f}% "
        f"bid_xlm={bid_amt:.2f} ask_xlm={ask_amt:.2f} "
        f"balUSDC={state.bal_usdc:.2f} balXLM={state.bal_xlm:.6f}"
    )

def report_line(state: State) -> str:
    prof = rolling_profit_usdc(state, ROLLING_HOURS)
    trades_rolling = sum(1 for t in state.trades if t.ts >= now_ts() - ROLLING_HOURS * 3600)
    return f"REPORT {ROLLING_HOURS}h trades={trades_rolling} profit={prof:.6f} {COUNTER_CODE}"


# -----------------------------
# Main loop
# -----------------------------

async def run() -> None:
    if not PAPER_TRADING:
        raise RuntimeError("This file is paper-trading only. PAPER_TRADING must stay True.")

    # Validate issuer for non-native counter assets
    if COUNTER_CODE.upper() != "XLM" and COUNTER_ISSUER.strip() == "":
        raise RuntimeError(
            "COUNTER_ISSUER is required for non-native counter asset (e.g., USDC). "
            "Set COUNTER_ISSUER in Railway variables."
        )

    state = load_state()

    last_report = 0.0

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                bid, ask, bid_amt, ask_amt = await fetch_best_bid_ask(session)
                spread = pct_spread(bid, ask)

                print(tick_line(bid, ask, bid_amt, ask_amt, state), flush=True)

                action = await simulate_cycle(state, bid, ask, bid_amt, ask_amt, spread)
                if action:
                    print(action, flush=True)

                # Housekeeping
                prune_old_trades(state)
                save_state(state)

                # Periodic report
                if now_ts() - last_report >= REPORT_EVERY_SECONDS:
                    print(report_line(state), flush=True)
                    last_report = now_ts()

                await asyncio.sleep(1)

            except aiohttp.ClientResponseError as e:
                print(f"HTTP error: {e.status} {e.message}", flush=True)
                await asyncio.sleep(3)
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(3)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
