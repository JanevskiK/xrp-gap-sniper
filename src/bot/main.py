#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# -------------------------
# Env helpers
# -------------------------

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v

def env_float(name: str, default: float) -> float:
    v = _env(name, "").strip().strip('"')
    return default if v == "" else float(v)

def env_int(name: str, default: int) -> int:
    v = _env(name, "").strip().strip('"')
    return default if v == "" else int(float(v))

def env_bool(name: str, default: bool = False) -> bool:
    v = _env(name, "").strip().strip('"').lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "y", "on")


# -------------------------
# Config (your variable names)
# -------------------------

HORIZON_URL = _env("HORIZON_URL", "https://horizon.stellar.org").strip().strip('"')

BASE_ASSET_TYPE = _env("BASE_ASSET_TYPE", "native").strip().strip('"')

COUNTER_ASSET_TYPE = _env("COUNTER_ASSET_TYPE", "credit_alphanum4").strip().strip('"')
COUNTER_ASSET_CODE = _env("COUNTER_ASSET_CODE", "USDC").strip().strip('"')
COUNTER_ASSET_ISSUER = _env("COUNTER_ASSET_ISSUER", "").strip().strip('"')

POLL_SECONDS = env_float("POLL_SECONDS", 1.0)

TRADE_COUNTER_AMOUNT = env_float("TRADE_COUNTER_AMOUNT", 100.0)  # USDC notional per side
MIN_SPREAD_PCT = env_float("MIN_SPREAD_PCT", 0.03)               # percent
MIN_DEPTH_MULT = env_float("MIN_DEPTH_MULT", 0.02)               # fraction of TRADE_COUNTER_AMOUNT required at top-of-book

PRINT_EVERY = env_int("PRINT_EVERY", 1)

INITIAL_USDC = env_float("INITIAL_USDC", 1000.0)
INITIAL_XLM = env_float("INITIAL_XLM", 0.0)

REQUOTE_SECONDS = env_int("REQUOTE_SECONDS", 60)
ORDER_TIMEOUT_SECONDS = env_int("ORDER_TIMEOUT_SECONDS", 900)

QUEUE_JOIN_FACTOR = env_float("QUEUE_JOIN_FACTOR", 1.0)
FILL_RATE_CAP = env_float("FILL_RATE_CAP", 999999.0)

NETWORK_FEE_XLM = env_float("NETWORK_FEE_XLM", 0.00001)
OPS_PER_FILL = env_int("OPS_PER_FILL", 2)

MAX_INVENTORY_XLM = env_float("MAX_INVENTORY_XLM", 2000.0)
MIN_INVENTORY_XLM = env_float("MIN_INVENTORY_XLM", 0.0)

STATS_FILE = _env("STATS_FILE", "/tmp/stellar_mm_trades.jsonl").strip().strip('"')

REPORT_EVERY_SECONDS = env_int("REPORT_EVERY_SECONDS", 300)
ROLLING_HOURS = env_int("ROLLING_HOURS", 48)

REQUOTE_BPS = env_int("REQUOTE_BPS", 10)

TARGET_XLM = env_float("TARGET_XLM", 500.0)
BAND_XLM = env_float("BAND_XLM", 150.0)

PAPER_TRADING = env_bool("PAPER_TRADING", True)


# -------------------------
# State / Trades
# -------------------------

@dataclass
class Trade:
    ts: float
    side: str                  # "BUY" or "SELL"
    price: float               # USDC per XLM
    xlm: float
    usdc: float                # BUY negative, SELL positive
    fee_xlm: float
    note: str = ""

@dataclass
class PaperOrder:
    created_ts: float
    side: str                  # "BUY" or "SELL"
    price: float               # limit price
    xlm_amount: float          # quantity
    remaining_xlm: float

@dataclass
class State:
    started_ts: float
    balUSDC: float
    balXLM: float
    trades: List[Trade]
    open_orders: List[PaperOrder]
    last_quote_ts: float
    last_report_ts: float
    last_print_ts: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_ts": self.started_ts,
            "balUSDC": self.balUSDC,
            "balXLM": self.balXLM,
            "trades": [asdict(t) for t in self.trades],
            "open_orders": [asdict(o) for o in self.open_orders],
            "last_quote_ts": self.last_quote_ts,
            "last_report_ts": self.last_report_ts,
            "last_print_ts": self.last_print_ts,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "State":
        trades = [Trade(**t) for t in d.get("trades", [])]
        open_orders = [PaperOrder(**o) for o in d.get("open_orders", [])]
        return State(
            started_ts=float(d.get("started_ts", time.time())),
            balUSDC=float(d.get("balUSDC", INITIAL_USDC)),
            balXLM=float(d.get("balXLM", INITIAL_XLM)),
            trades=trades,
            open_orders=open_orders,
            last_quote_ts=float(d.get("last_quote_ts", 0.0)),
            last_report_ts=float(d.get("last_report_ts", 0.0)),
            last_print_ts=float(d.get("last_print_ts", 0.0)),
        )


def load_state() -> State:
    if not os.path.exists(STATS_FILE):
        return State(time.time(), INITIAL_USDC, INITIAL_XLM, [], [], 0.0, 0.0, 0.0)
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return State(time.time(), INITIAL_USDC, INITIAL_XLM, [], [], 0.0, 0.0, 0.0)
            return State.from_dict(json.loads(txt))
    except Exception:
        return State(time.time(), INITIAL_USDC, INITIAL_XLM, [], [], 0.0, 0.0, 0.0)

def save_state(state: State) -> None:
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


# -------------------------
# Horizon order book
# -------------------------

def _asset_params(prefix: str, asset_type: str, code: str, issuer: str) -> Dict[str, str]:
    if asset_type == "native":
        return {f"{prefix}_asset_type": "native"}
    if code.strip() == "" or issuer.strip() == "":
        raise RuntimeError(f"Missing {prefix} asset code/issuer for non-native asset.")
    return {
        f"{prefix}_asset_type": asset_type,
        f"{prefix}_asset_code": code,
        f"{prefix}_asset_issuer": issuer,
    }

async def fetch_best_bid_ask(session: aiohttp.ClientSession) -> Tuple[float, float, float, float]:
    # selling = XLM (native), buying = USDC (credit)
    url = f"{HORIZON_URL.rstrip('/')}/order_book"
    params: Dict[str, str] = {}
    params.update(_asset_params("selling", BASE_ASSET_TYPE, "", ""))
    params.update(_asset_params("buying", COUNTER_ASSET_TYPE, COUNTER_ASSET_CODE, COUNTER_ASSET_ISSUER))

    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        data = await r.json()

    bids = data.get("bids", [])
    asks = data.get("asks", [])
    if not bids or not asks:
        raise RuntimeError("Orderbook empty (no bids or asks).")

    bid = float(bids[0]["price"])
    bid_xlm = float(bids[0]["amount"])
    ask = float(asks[0]["price"])
    ask_xlm = float(asks[0]["amount"])
    return bid, ask, bid_xlm, ask_xlm

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    return (ask - bid) / mid * 100.0


# -------------------------
# Profit metrics
# -------------------------

def rolling_profit_usdc(trades: List[Trade], hours: int) -> Tuple[float, int]:
    cutoff = time.time() - hours * 3600
    p = 0.0
    n = 0
    for t in trades:
        if t.ts >= cutoff:
            p += t.usdc
            n += 1
    return p, n

def portfolio_value_usdc(bal_usdc: float, bal_xlm: float, bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return bal_usdc + bal_xlm * mid

def prune_trades(state: State, keep_hours: int = 24 * 14) -> None:
    cutoff = time.time() - keep_hours * 3600
    state.trades = [t for t in state.trades if t.ts >= cutoff]


# -------------------------
# Market-maker behavior (paper)
# -------------------------

def inventory_skew(state: State) -> float:
    """
    Returns a multiplier that slightly biases quoting sizes
    depending on how far you are from TARGET_XLM.
    """
    if BAND_XLM <= 0:
        return 1.0
    delta = state.balXLM - TARGET_XLM
    # If you have too much XLM, reduce buys; if too little XLM, reduce sells.
    # Keep it mild.
    skew = max(-1.0, min(1.0, delta / BAND_XLM))
    return skew  # -1..+1

def cancel_all_orders(state: State) -> None:
    state.open_orders = []

def place_quotes_if_ok(state: State, bid: float, ask: float, bid_xlm: float, ask_xlm: float, spr: float) -> None:
    # Only quote when spread is large enough
    if spr < MIN_SPREAD_PCT:
        cancel_all_orders(state)
        return

    # Depth check at top of book (in USDC terms)
    min_needed = TRADE_COUNTER_AMOUNT * MIN_DEPTH_MULT
    if bid_xlm * bid < min_needed or ask_xlm * ask < min_needed:
        cancel_all_orders(state)
        return

    # Requote cadence
    now = time.time()
    if now - state.last_quote_ts < REQUOTE_SECONDS and state.open_orders:
        return

    # Clear old quotes and place new ones at current best bid/ask
    cancel_all_orders(state)

    # Decide notional per side, then convert to XLM
    buy_usdc = min(TRADE_COUNTER_AMOUNT, state.balUSDC)
    buy_xlm = buy_usdc / bid if bid > 0 else 0.0

    sell_xlm = min(TRADE_COUNTER_AMOUNT / ask if ask > 0 else 0.0, state.balXLM)

    # Inventory controls
    if state.balXLM + buy_xlm > MAX_INVENTORY_XLM:
        buy_xlm = max(0.0, MAX_INVENTORY_XLM - state.balXLM)
    if state.balXLM - sell_xlm < MIN_INVENTORY_XLM:
        sell_xlm = max(0.0, state.balXLM - MIN_INVENTORY_XLM)

    # Mild skew based on inventory
    skew = inventory_skew(state)
    if skew > 0:        # too much XLM => reduce buys
        buy_xlm *= (1.0 - 0.35 * skew)
    elif skew < 0:      # too little XLM => reduce sells
        sell_xlm *= (1.0 + 0.35 * skew)  # skew negative => reduces

    buy_xlm = max(0.0, buy_xlm)
    sell_xlm = max(0.0, sell_xlm)

    if buy_xlm > 0:
        state.open_orders.append(PaperOrder(now, "BUY", bid, buy_xlm, buy_xlm))
    if sell_xlm > 0:
        state.open_orders.append(PaperOrder(now, "SELL", ask, sell_xlm, sell_xlm))

    state.last_quote_ts = now

def simulate_fills(state: State, bid: float, ask: float, bid_xlm: float, ask_xlm: float) -> List[str]:
    """
    Paper fill model:
    - BUY limit at bid fills when market ask <= our bid (rare at top-of-book),
      so we simulate partial fills based on bid-side activity instead.
    - SELL limit at ask fills when market bid >= our ask (rare),
      so we simulate partial fills based on ask-side activity instead.

    This is a simplified maker simulation: the fill rate is proportional to top depth
    and QUEUE_JOIN_FACTOR and POLL_SECONDS.
    """
    now = time.time()
    notes: List[str] = []

    # Expire orders
    still_open: List[PaperOrder] = []
    for o in state.open_orders:
        if now - o.created_ts > ORDER_TIMEOUT_SECONDS:
            continue
        still_open.append(o)
    state.open_orders = still_open

    if not state.open_orders:
        return notes

    # “Fill fraction” per tick:
    # increase with depth, but capped. queue factor reduces or increases.
    # This isn’t perfect market reality, but it behaves like the earlier “spread capture” sim.
    def fill_fraction(depth_xlm: float) -> float:
        # base fraction depends on depth vs our notional, scaled by polling cadence
        base = min(1.0, (depth_xlm / max(1e-9, (TRADE_COUNTER_AMOUNT / max(1e-9, ((bid+ask)/2))))) * 0.05)
        # queue join factor: >1 means you join later => slower fills
        q = max(0.2, min(5.0, QUEUE_JOIN_FACTOR))
        frac = (base / q) * max(0.2, min(2.0, POLL_SECONDS))
        return max(0.0, min(0.25, frac))  # cap 25% per tick

    buy_fill_frac = fill_fraction(bid_xlm)
    sell_fill_frac = fill_fraction(ask_xlm)

    # Respect FILL_RATE_CAP by limiting xlm per tick
    cap_xlm = FILL_RATE_CAP if FILL_RATE_CAP > 0 else 999999.0

    total_fee_xlm = NETWORK_FEE_XLM * OPS_PER_FILL

    new_orders: List[PaperOrder] = []

    for o in state.open_orders:
        if o.remaining_xlm <= 0:
            continue

        if o.side == "BUY":
            # fill some of remaining
            xlm_fill = min(o.remaining_xlm, o.remaining_xlm * buy_fill_frac, cap_xlm)
            # Also require we actually have USDC
            usdc_cost = xlm_fill * o.price
            if usdc_cost > state.balUSDC:
                # cannot fill more than we can pay
                xlm_fill = max(0.0, state.balUSDC / max(1e-9, o.price))
                usdc_cost = xlm_fill * o.price

            if xlm_fill > 0:
                state.balUSDC -= usdc_cost
                state.balXLM += xlm_fill
                state.balXLM = max(0.0, state.balXLM - (total_fee_xlm / 2))

                state.trades.append(Trade(
                    ts=now,
                    side="BUY",
                    price=o.price,
                    xlm=xlm_fill,
                    usdc=-usdc_cost,
                    fee_xlm=total_fee_xlm / 2,
                    note="paper_fill",
                ))
                o.remaining_xlm -= xlm_fill
                notes.append(f"FILL BUY xlm={xlm_fill:.6f} @ {o.price:.7f}")

        else:  # SELL
            xlm_fill = min(o.remaining_xlm, o.remaining_xlm * sell_fill_frac, cap_xlm)
            if xlm_fill > state.balXLM:
                xlm_fill = max(0.0, state.balXLM)

            if xlm_fill > 0:
                usdc_get = xlm_fill * o.price
                state.balXLM -= xlm_fill
                state.balUSDC += usdc_get
                state.balXLM = max(0.0, state.balXLM - (total_fee_xlm / 2))

                state.trades.append(Trade(
                    ts=now,
                    side="SELL",
                    price=o.price,
                    xlm=xlm_fill,
                    usdc=usdc_get,
                    fee_xlm=total_fee_xlm / 2,
                    note="paper_fill",
                ))
                o.remaining_xlm -= xlm_fill
                notes.append(f"FILL SELL xlm={xlm_fill:.6f} @ {o.price:.7f}")

        if o.remaining_xlm > 1e-9:
            new_orders.append(o)

    state.open_orders = new_orders
    return notes


# -------------------------
# Logging
# -------------------------

def fmt_tick(bid: float, ask: float, bid_xlm: float, ask_xlm: float, state: State) -> str:
    spr = spread_pct(bid, ask)
    return (
        f"TICK bid={bid:.7f} ask={ask:.7f} spread={spr:.4f}% "
        f"bid_xlm={bid_xlm:.2f} ask_xlm={ask_xlm:.2f} "
        f"balUSDC={state.balUSDC:.2f} balXLM={state.balXLM:.6f}"
    )

def fmt_report(state: State, bid: float, ask: float) -> str:
    p48, n48 = rolling_profit_usdc(state.trades, ROLLING_HOURS)
    total_val = portfolio_value_usdc(state.balUSDC, state.balXLM, bid, ask)
    initial_val = INITIAL_USDC + INITIAL_XLM * ((bid + ask) / 2.0)
    p_total = total_val - initial_val
    return (
        f"REPORT {ROLLING_HOURS}h trades={n48} profit48h={p48:.6f} {COUNTER_ASSET_CODE} | "
        f"total_value={total_val:.2f} {COUNTER_ASSET_CODE} | profit_total={p_total:.2f} {COUNTER_ASSET_CODE} | "
        f"open_orders={len(state.open_orders)} requote_bps={REQUOTE_BPS}"
    )


# -------------------------
# Main loop
# -------------------------

async def run() -> None:
    if not PAPER_TRADING:
        raise RuntimeError("This file is paper-trading only. Set PAPER_TRADING=1.")

    if BASE_ASSET_TYPE != "native":
        raise RuntimeError("Expected BASE_ASSET_TYPE=native.")
    if COUNTER_ASSET_ISSUER.strip() == "":
        raise RuntimeError("COUNTER_ASSET_ISSUER is required (USDC issuer).")

    state = load_state()

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            now = time.time()
            try:
                bid, ask, bid_xlm, ask_xlm = await fetch_best_bid_ask(session)
                spr = spread_pct(bid, ask)

                # Print tick
                if now - state.last_print_ts >= max(1, PRINT_EVERY):
                    print(fmt_tick(bid, ask, bid_xlm, ask_xlm, state), flush=True)
                    state.last_print_ts = now

                # Quote management
                place_quotes_if_ok(state, bid, ask, bid_xlm, ask_xlm, spr)

                # Fill simulation
                fill_notes = simulate_fills(state, bid, ask, bid_xlm, ask_xlm)
                for n in fill_notes:
                    print(n, flush=True)

                prune_trades(state)
                save_state(state)

                # Report
                if now - state.last_report_ts >= max(1, REPORT_EVERY_SECONDS):
                    print(fmt_report(state, bid, ask), flush=True)
                    state.last_report_ts = now

                await asyncio.sleep(max(0.1, POLL_SECONDS))

            except aiohttp.ClientResponseError as e:
                print(f"HTTP error: {e.status} {e.message}", flush=True)
                await asyncio.sleep(2)
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(2)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

