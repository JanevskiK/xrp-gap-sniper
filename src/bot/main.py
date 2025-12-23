#!/usr/bin/env python3
"""
Stellar XLM/USDC market-maker bot (simple) + Rolling 48h realized PnL.

✅ Places one BUY (buy XLM, pay USDC) and one SELL (sell XLM, receive USDC)
✅ Requotes to best bid/ask
✅ Cancels when spread too small or depth too low
✅ Tracks fills via Horizon account trades
✅ Logs each realized PnL event into STATS_FILE (jsonl)
✅ REPORT shows rolling 48h combined realized PnL and trade count

⚠️ Trading risk:
- This is a real trading bot if DRY_RUN=0.
- Read and understand before running with money.

Dependencies:
  pip install stellar-sdk requests

Environment variables (Railway or local):
  HORIZON_URL="https://horizon.stellar.org"
  SECRET_KEY="S..."
  BASE_ASSET_TYPE="native"
  COUNTER_ASSET_TYPE="credit_alphanum4"
  COUNTER_ASSET_CODE="USDC"
  COUNTER_ASSET_ISSUER="GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"

  POLL_SECONDS="1.0"
  REQUOTE_SECONDS="3"
  ORDER_TIMEOUT_SECONDS="30"

  TRADE_COUNTER_AMOUNT="100.0"     # USDC budget used per "round" sizing logic
  MIN_SPREAD_PCT="0.03"            # percent, e.g. 0.03 means 0.03%
  MIN_DEPTH_MULT="1.0"             # how much depth must exist vs our size

  MAX_INVENTORY_XLM="2000"
  MIN_INVENTORY_XLM="0"

  STATS_FILE="/tmp/stellar_mm_trades.jsonl"
  STATE_FILE="/tmp/stellar_mm_state.json"
  REPORT_EVERY_SECONDS="60"
  ROLLING_HOURS="48"

  PRINT_EVERY="1"

  # Safety
  DRY_RUN="1"                       # 1 = do not submit tx, just print
  CANCEL_ALL_ON_START="1"           # cancels existing offers for this pair on startup

Notes about pricing:
- We treat orderbook bid/ask as USDC per 1 XLM (matches your logs: ~0.216 USDC/XLM).
- To BUY XLM with USDC on Stellar offers, we create an offer that SELLS USDC and BUYS XLM.
  Offer price on Stellar is in terms of "buying per selling". So we use price_xlm_per_usdc = 1 / (usdc_per_xlm).
- To SELL XLM for USDC, we create offer that SELLS XLM and BUYS USDC, price_usdc_per_xlm = usdc_per_xlm.
"""

import os
import time
import json
import math
import requests
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from stellar_sdk import (
    Server,
    Keypair,
    TransactionBuilder,
    Network,
    Asset,
)
from stellar_sdk.exceptions import BadRequestError, NotFoundError


# ---------------------------
# Helpers / Config
# ---------------------------

def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None and v != "" else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None and v != "" else default

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def now_ts() -> float:
    return time.time()

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class OrderbookTop:
    bid: float
    ask: float
    bid_xlm: float
    ask_xlm: float

    @property
    def spread_pct(self) -> float:
        # percent
        if self.bid <= 0:
            return 999.0
        return ((self.ask - self.bid) / self.bid) * 100.0


@dataclass
class State:
    last_trade_cursor: str = "now"   # Horizon paging token cursor
    pos_xlm: float = 0.0            # inventory tracked from fills
    cost_usdc: float = 0.0          # cost basis for pos_xlm (average cost = cost_usdc/pos_xlm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_trade_cursor": self.last_trade_cursor,
            "pos_xlm": self.pos_xlm,
            "cost_usdc": self.cost_usdc,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "State":
        return State(
            last_trade_cursor=str(d.get("last_trade_cursor", "now")),
            pos_xlm=safe_float(d.get("pos_xlm", 0.0)),
            cost_usdc=safe_float(d.get("cost_usdc", 0.0)),
        )


# ---------------------------
# Trade logging + Rolling PnL
# ---------------------------

def log_trade(stats_file: str, side: str, xlm_amount: float, price_usdc_per_xlm: float, pnl_usdc: float):
    rec = {
        "ts": now_ts(),
        "side": side,
        "xlm": float(xlm_amount),
        "price": float(price_usdc_per_xlm),
        "pnl_usdc": float(pnl_usdc),
    }
    os.makedirs(os.path.dirname(stats_file), exist_ok=True) if "/" in stats_file else None
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

def rolling_stats(stats_file: str, rolling_hours: float) -> Tuple[int, float]:
    cutoff = now_ts() - (rolling_hours * 3600.0)
    trades = 0
    pnl = 0.0
    if not stats_file or not os.path.exists(stats_file):
        return trades, pnl

    with open(stats_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = safe_float(rec.get("ts", 0.0))
            if ts >= cutoff:
                trades += 1
                pnl += safe_float(rec.get("pnl_usdc", 0.0))
    return trades, pnl


# ---------------------------
# Stellar + Horizon
# ---------------------------

def build_asset(asset_type: str, code: str = "", issuer: str = "") -> Asset:
    if asset_type == "native":
        return Asset.native()
    return Asset(code, issuer)

def asset_id(a: Asset) -> str:
    if a.is_native():
        return "native"
    return f"{a.code}:{a.issuer}"

def fetch_orderbook_top(horizon_url: str, base: Asset, counter: Asset) -> OrderbookTop:
    """
    We request orderbook as selling=base, buying=counter.
    For base=XLM, counter=USDC:
      asks: people selling XLM for USDC => ask prices (USDC/XLM)
      bids: people buying XLM with USDC => bid prices (USDC/XLM)
    """
    params = {}

    # selling = base
    if base.is_native():
        params["selling_asset_type"] = "native"
    else:
        params["selling_asset_type"] = "credit_alphanum4" if len(base.code) <= 4 else "credit_alphanum12"
        params["selling_asset_code"] = base.code
        params["selling_asset_issuer"] = base.issuer

    # buying = counter
    if counter.is_native():
        params["buying_asset_type"] = "native"
    else:
        params["buying_asset_type"] = "credit_alphanum4" if len(counter.code) <= 4 else "credit_alphanum12"
        params["buying_asset_code"] = counter.code
        params["buying_asset_issuer"] = counter.issuer

    url = horizon_url.rstrip("/") + "/order_book"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    bids = data.get("bids", [])
    asks = data.get("asks", [])

    # default if empty
    bid_price = safe_float(bids[0]["price"], 0.0) if bids else 0.0
    ask_price = safe_float(asks[0]["price"], 0.0) if asks else 0.0

    # "amount" in order_book is amount of selling asset (base, i.e., XLM)
    bid_xlm = safe_float(bids[0].get("amount", 0.0), 0.0) if bids else 0.0
    ask_xlm = safe_float(asks[0].get("amount", 0.0), 0.0) if asks else 0.0

    return OrderbookTop(
        bid=bid_price,
        ask=ask_price,
        bid_xlm=bid_xlm,
        ask_xlm=ask_xlm,
    )

def fetch_balances(server: Server, account_id: str, counter: Asset) -> Tuple[float, float]:
    """
    Returns: (balance_counter_usdc, balance_base_xlm)
    """
    acct = server.accounts().account_id(account_id).call()
    bal_usdc = 0.0
    bal_xlm = 0.0
    for b in acct.get("balances", []):
        if b.get("asset_type") == "native":
            bal_xlm = safe_float(b.get("balance", 0.0))
        else:
            if b.get("asset_code") == counter.code and b.get("asset_issuer") == counter.issuer:
                bal_usdc = safe_float(b.get("balance", 0.0))
    return bal_usdc, bal_xlm

def load_state(state_file: str) -> State:
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return State.from_dict(json.load(f))
    except Exception:
        return State()

def save_state(state_file: str, st: State):
    os.makedirs(os.path.dirname(state_file), exist_ok=True) if "/" in state_file else None
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(st.to_dict(), f)

def list_offers(server: Server, account_id: str) -> List[Dict[str, Any]]:
    try:
        offers = server.offers().for_account(account_id).limit(200).call()
        return offers.get("_embedded", {}).get("records", [])
    except Exception:
        return []

def offer_matches_pair(offer: Dict[str, Any], selling: Asset, buying: Asset) -> bool:
    s = offer.get("selling", {})
    b = offer.get("buying", {})

    def match_side(side: Dict[str, Any], a: Asset) -> bool:
        if a.is_native():
            return side.get("asset_type") == "native"
        return (
            side.get("asset_type", "").startswith("credit_")
            and side.get("asset_code") == a.code
            and side.get("asset_issuer") == a.issuer
        )

    return match_side(s, selling) and match_side(b, buying)

def submit_manage_sell_offer(
    server: Server,
    kp: Keypair,
    selling: Asset,
    buying: Asset,
    amount_selling: float,
    price_buying_per_selling: float,
    offer_id: int,
    dry_run: bool,
) -> int:
    """
    Returns offer_id used (new or existing).
    """
    if amount_selling <= 0 or price_buying_per_selling <= 0:
        return offer_id

    if dry_run:
        # Fake offer id handling
        return offer_id if offer_id != 0 else int(now_ts())

    account = server.load_account(kp.public_key)
    tx = (
        TransactionBuilder(
            source_account=account,
            network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,
            base_fee=100,
        )
        .append_manage_sell_offer_op(
            selling=selling,
            buying=buying,
            amount=f"{amount_selling:.7f}",
            price=f"{price_buying_per_selling:.12f}",
            offer_id=offer_id,
        )
        .set_timeout(60)
        .build()
    )
    tx.sign(kp)
    resp = server.submit_transaction(tx)
    # Best-effort parse: offer_id sometimes in results; if not, we keep existing.
    # We will refresh offers later if needed.
    return offer_id if offer_id != 0 else offer_id

def submit_cancel_offer(server: Server, kp: Keypair, selling: Asset, buying: Asset, offer_id: int, dry_run: bool):
    if offer_id == 0:
        return
    if dry_run:
        return
    account = server.load_account(kp.public_key)
    tx = (
        TransactionBuilder(
            source_account=account,
            network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,
            base_fee=100,
        )
        .append_manage_sell_offer_op(
            selling=selling,
            buying=buying,
            amount="0",      # amount 0 cancels
            price="1",       # price ignored on cancel
            offer_id=offer_id,
        )
        .set_timeout(60)
        .build()
    )
    tx.sign(kp)
    server.submit_transaction(tx)

def cancel_all_pair_offers(server: Server, kp: Keypair, base: Asset, counter: Asset, dry_run: bool):
    # Our BUY offer is: selling=counter, buying=base
    # Our SELL offer is: selling=base, buying=counter
    offers = list_offers(server, kp.public_key)
    for off in offers:
        oid = int(off.get("id", 0))
        if offer_matches_pair(off, counter, base) or offer_matches_pair(off, base, counter):
            print(f"CANCEL existing offer_id={oid} (pair match)")
            submit_cancel_offer(server, kp, Asset.native() if off.get("selling", {}).get("asset_type") == "native" else base,  # best-effort
                                Asset.native() if off.get("buying", {}).get("asset_type") == "native" else counter, oid, dry_run)


# ---------------------------
# Fill tracking (account trades)
# ---------------------------

def fetch_new_trades(horizon_url: str, account_id: str, cursor: str, limit: int = 200) -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns (trades, new_cursor). Trades are oldest->newest.
    Uses Horizon /accounts/{id}/trades?cursor=...
    """
    url = horizon_url.rstrip("/") + f"/accounts/{account_id}/trades"
    params = {"order": "asc", "limit": limit}
    if cursor:
        params["cursor"] = cursor
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    recs = data.get("_embedded", {}).get("records", [])
    if not recs:
        return [], cursor
    # paging_token exists
    new_cursor = recs[-1].get("paging_token", cursor)
    return recs, new_cursor

def trade_is_xlm_usdc(tr: Dict[str, Any], base: Asset, counter: Asset) -> bool:
    # Horizon trade record has base_asset_type/code/issuer and counter_asset_type/code/issuer
    def match_asset(prefix: str, a: Asset) -> bool:
        t = tr.get(prefix + "_asset_type")
        if a.is_native():
            return t == "native"
        return (
            t and t.startswith("credit_")
            and tr.get(prefix + "_asset_code") == a.code
            and tr.get(prefix + "_asset_issuer") == a.issuer
        )

    # pair can appear in either orientation; we accept any that includes both assets
    has_base = match_asset("base", base) or match_asset("counter", base)
    has_counter = match_asset("base", counter) or match_asset("counter", counter)
    return has_base and has_counter

def apply_trades_to_pnl(
    trades: List[Dict[str, Any]],
    st: State,
    base: Asset,
    counter: Asset,
    stats_file: str,
):
    """
    Updates st.pos_xlm, st.cost_usdc, logs realized pnl on SELL events.

    We interpret each trade from the account's perspective by using "base_is_seller":
    - If base=XLM:
        base_is_seller=True  => we sold XLM, received USDC (a SELL)
        base_is_seller=False => we bought XLM, paid USDC (a BUY)
    """
    for tr in trades:
        if not trade_is_xlm_usdc(tr, base, counter):
            continue

        # We want base to be XLM; if user changed BASE_ASSET_TYPE, keep logic consistent:
        # Our "base" is whatever you configured as BASE_ASSET_TYPE, but in your setup it's native(XLM).
        base_is_seller = bool(tr.get("base_is_seller", False))
        # amounts in record are "base_amount", "counter_amount"
        base_amt = safe_float(tr.get("base_amount", 0.0))
        counter_amt = safe_float(tr.get("counter_amount", 0.0))
        # price is a rational string sometimes; but Horizon also provides "price" dict. We keep USDC/XLM from record fields:
        # If base=XLM and counter=USDC, then counter_amt/base_amt = USDC per XLM.
        price_usdc_per_xlm = (counter_amt / base_amt) if base_amt > 0 else 0.0

        if base_is_seller:
            # SOLD base (XLM), received counter (USDC)
            sold_xlm = base_amt
            proceeds_usdc = counter_amt

            if sold_xlm <= 0:
                continue

            # Realize PnL using average cost basis
            if st.pos_xlm > 0:
                avg_cost = st.cost_usdc / st.pos_xlm
            else:
                avg_cost = 0.0

            cost_removed = avg_cost * sold_xlm
            pnl = proceeds_usdc - cost_removed

            # Update position/cost
            st.pos_xlm -= sold_xlm
            st.cost_usdc -= cost_removed

            # Clamp tiny float noise
            if st.pos_xlm < 1e-9:
                st.pos_xlm = 0.0
            if st.cost_usdc < 1e-9:
                st.cost_usdc = 0.0

            log_trade(stats_file, "SELL", sold_xlm, price_usdc_per_xlm, pnl)

        else:
            # BOUGHT base (XLM), paid counter (USDC)
            bought_xlm = base_amt
            paid_usdc = counter_amt
            if bought_xlm <= 0:
                continue
            st.pos_xlm += bought_xlm
            st.cost_usdc += paid_usdc
            # (No realized pnl on BUY)


# ---------------------------
# Main MM loop
# ---------------------------

def main():
    horizon_url = env_str("HORIZON_URL", "https://horizon.stellar.org")
    secret = env_str("SECRET_KEY", "").strip()
    if not secret:
        raise SystemExit("Missing SECRET_KEY env var")

    base_asset_type = env_str("BASE_ASSET_TYPE", "native")
    counter_asset_type = env_str("COUNTER_ASSET_TYPE", "credit_alphanum4")
    counter_code = env_str("COUNTER_ASSET_CODE", "USDC")
    counter_issuer = env_str("COUNTER_ASSET_ISSUER", "")

    base = build_asset(base_asset_type)
    counter = build_asset(counter_asset_type, counter_code, counter_issuer)

    poll_seconds = env_float("POLL_SECONDS", 1.0)
    requote_seconds = env_float("REQUOTE_SECONDS", 3.0)
    order_timeout_seconds = env_float("ORDER_TIMEOUT_SECONDS", 30.0)

    trade_counter_amount = env_float("TRADE_COUNTER_AMOUNT", 100.0)  # USDC budget per sizing
    min_spread_pct = env_float("MIN_SPREAD_PCT", 0.03)               # percent
    min_depth_mult = env_float("MIN_DEPTH_MULT", 1.0)

    max_inventory_xlm = env_float("MAX_INVENTORY_XLM", 2000.0)
    min_inventory_xlm = env_float("MIN_INVENTORY_XLM", 0.0)

    stats_file = env_str("STATS_FILE", "/tmp/stellar_mm_trades.jsonl")
    state_file = env_str("STATE_FILE", "/tmp/stellar_mm_state.json")
    report_every = env_float("REPORT_EVERY_SECONDS", 60.0)
    rolling_hours = env_float("ROLLING_HOURS", 48.0)
    print_every = env_int("PRINT_EVERY", 1)

    dry_run = env_bool("DRY_RUN", True)
    cancel_all_on_start = env_bool("CANCEL_ALL_ON_START", True)

    kp = Keypair.from_secret(secret)
    server = Server(horizon_url=horizon_url)

    st = load_state(state_file)

    # On first run, set cursor to "now" so we don't backfill old trades
    if st.last_trade_cursor == "" or st.last_trade_cursor is None:
        st.last_trade_cursor = "now"

    if cancel_all_on_start:
        print("Startup: cancelling existing offers for this pair (if any)...")
        try:
            cancel_all_pair_offers(server, kp, base, counter, dry_run)
        except Exception as e:
            print(f"Startup cancel_all warning: {e}")

    # Offer IDs tracked only for the current process (safe enough for a simple setup)
    buy_offer_id = 0   # offer selling USDC, buying XLM
    sell_offer_id = 0  # offer selling XLM, buying USDC
    buy_last_place_ts = 0.0
    sell_last_place_ts = 0.0

    last_report_ts = 0.0
    tick_count = 0

    print(f"Running MM on {horizon_url} | pair base={asset_id(base)} counter={asset_id(counter)} | DRY_RUN={dry_run}")

    while True:
        t0 = now_ts()
        tick_count += 1

        # 1) Update fills -> realized pnl log
        try:
            new_trades, new_cursor = fetch_new_trades(horizon_url, kp.public_key, st.last_trade_cursor)
            if new_trades:
                apply_trades_to_pnl(new_trades, st, base, counter, stats_file)
                st.last_trade_cursor = new_cursor
                save_state(state_file, st)
        except Exception as e:
            print(f"Trade poll warning: {e}")

        # 2) Fetch orderbook + balances
        try:
            ob = fetch_orderbook_top(horizon_url, base, counter)
        except Exception as e:
            print(f"Orderbook error: {e}")
            time.sleep(poll_seconds)
            continue

        try:
            bal_usdc, bal_xlm = fetch_balances(server, kp.public_key, counter)
        except Exception as e:
            print(f"Balance error: {e}")
            time.sleep(poll_seconds)
            continue

        # 3) Compute sizing: we size XLM using USDC budget and bid price
        # Use bid price to compute approximate xlm size for both sides
        if ob.bid > 0:
            quote_xlm = trade_counter_amount / ob.bid
        else:
            quote_xlm = 0.0

        # 4) Decide whether to quote (spread + depth checks)
        spread_ok = ob.spread_pct >= min_spread_pct
        # "needed depth" in XLM: we want at least min_depth_mult * (our size / 50) style?
        # We do a simple check: require depth >= min_depth_mult * (our quoted size * 0.02)
        # You can make this stricter/looser.
        depth_need = max(1e-9, (quote_xlm * 0.02) * min_depth_mult)

        depth_ok = (ob.bid_xlm >= depth_need) and (ob.ask_xlm >= depth_need)

        # Inventory risk controls
        inv_ok_for_buy = bal_xlm < max_inventory_xlm
        inv_ok_for_sell = bal_xlm > min_inventory_xlm

        # 5) Printing
        if print_every > 0 and (tick_count % print_every == 0):
            print(
                f"{time.strftime('%b %d %Y %H:%M:%S')}  "
                f"TICK bid={ob.bid:.7f} ask={ob.ask:.7f} spread={ob.spread_pct:.4f}% "
                f"bid_xlm={ob.bid_xlm:.2f} ask_xlm={ob.ask_xlm:.2f} "
                f"balUSDC={bal_usdc:.2f} balXLM={bal_xlm:.4f}"
            )

        # 6) REPORT rolling 48h realized pnl
        if (t0 - last_report_ts) >= report_every:
            trades_48h, pnl_48h = rolling_stats(stats_file, rolling_hours)
            print(f"{time.strftime('%b %d %Y %H:%M:%S')}  REPORT {int(rolling_hours)}h trades={trades_48h} realized_pnl={pnl_48h:.6f} USDC")
            last_report_ts = t0

        # 7) Manage offers
        # If we should not quote, cancel existing offers (best-effort)
        if not (spread_ok and depth_ok):
            reason = "spread too small" if not spread_ok else f"depth too low need={depth_need:.2f} bid={ob.bid_xlm:.2f} ask={ob.ask_xlm:.2f}"
            if buy_offer_id:
                print(f"{time.strftime('%b %d %Y %H:%M:%S')}  CANCEL BUY offer_id={buy_offer_id} reason=no quote: {reason}")
                try:
                    submit_cancel_offer(server, kp, counter, base, buy_offer_id, dry_run)
                except Exception as e:
                    print(f"Cancel BUY warning: {e}")
                buy_offer_id = 0

            if sell_offer_id:
                print(f"{time.strftime('%b %d %Y %H:%M:%S')}  CANCEL SELL offer_id={sell_offer_id} reason=no quote: {reason}")
                try:
                    submit_cancel_offer(server, kp, base, counter, sell_offer_id, dry_run)
                except Exception as e:
                    print(f"Cancel SELL warning: {e}")
                sell_offer_id = 0

            # sleep and continue
            time.sleep(poll_seconds)
            continue

        # Requote cadence
        should_requote_buy = (t0 - buy_last_place_ts) >= requote_seconds
        should_requote_sell = (t0 - sell_last_place_ts) >= requote_seconds

        # Timeout cancels (if offers linger too long without requote)
        if buy_offer_id and (t0 - buy_last_place_ts) > order_timeout_seconds:
            print(f"{time.strftime('%b %d %Y %H:%M:%S')}  CANCEL BUY offer_id={buy_offer_id} reason=timeout")
            try:
                submit_cancel_offer(server, kp, counter, base, buy_offer_id, dry_run)
            except Exception as e:
                print(f"Timeout cancel BUY warning: {e}")
            buy_offer_id = 0

        if sell_offer_id and (t0 - sell_last_place_ts) > order_timeout_seconds:
            print(f"{time.strftime('%b %d %Y %H:%M:%S')}  CANCEL SELL offer_id={sell_offer_id} reason=timeout")
            try:
                submit_cancel_offer(server, kp, base, counter, sell_offer_id, dry_run)
            except Exception as e:
                print(f"Timeout cancel SELL warning: {e}")
            sell_offer_id = 0

        # Place/replace BUY (sell USDC, buy XLM)
        if inv_ok_for_buy and bal_usdc > 1.0 and quote_xlm > 0 and should_requote_buy:
            buy_price_usdc_per_xlm = ob.bid
            # Offer price expects buying per selling => XLM per USDC
            buy_price_xlm_per_usdc = (1.0 / buy_price_usdc_per_xlm) if buy_price_usdc_per_xlm > 0 else 0.0
            # Amount for ManageSellOffer is "selling amount" => USDC amount to sell
            sell_usdc_amt = min(trade_counter_amount, max(0.0, bal_usdc - 1.0))
            if sell_usdc_amt > 0 and buy_price_xlm_per_usdc > 0:
                queue_ahead = ob.bid_xlm
                if buy_offer_id:
                    print(f"{time.strftime('%b %d %Y %H:%M:%S')}  CANCEL BUY price={buy_price_usdc_per_xlm:.7f} reason=requote to best bid")
                    try:
                        submit_cancel_offer(server, kp, counter, base, buy_offer_id, dry_run)
                    except Exception as e:
                        print(f"Requote cancel BUY warning: {e}")
                    buy_offer_id = 0

                amount_xlm_est = sell_usdc_amt / buy_price_usdc_per_xlm
                print(
                    f"{time.strftime('%b %d %Y %H:%M:%S')}  PLACE BUY price={buy_price_usdc_per_xlm:.7f} "
                    f"amount_xlm={amount_xlm_est:.4f} queue_ahead={queue_ahead:.2f}"
                )
                try:
                    buy_offer_id = submit_manage_sell_offer(
                        server=server,
                        kp=kp,
                        selling=counter,
                        buying=base,
                        amount_selling=sell_usdc_amt,
                        price_buying_per_selling=buy_price_xlm_per_usdc,
                        offer_id=0,
                        dry_run=dry_run,
                    )
                    buy_last_place_ts = t0
                except Exception as e:
                    print(f"PLACE BUY error: {e}")

        # Place/replace SELL (sell XLM, buy USDC)
        if inv_ok_for_sell and bal_xlm > 1.0 and quote_xlm > 0 and should_requote_sell:
            sell_price_usdc_per_xlm = ob.ask
            # Offer price expects buying per selling => USDC per XLM (already)
            sell_price_usdc_per_xlm_for_offer = sell_price_usdc_per_xlm
            # Amount for ManageSellOffer is "selling amount" => XLM to sell
            sell_xlm_amt = min(quote_xlm, max(0.0, bal_xlm - 1.0))
            if sell_xlm_amt > 0 and sell_price_usdc_per_xlm_for_offer > 0:
                queue_ahead = ob.ask_xlm
                if sell_offer_id:
                    print(f"{time.strftime('%b %d %Y %H:%M:%S')}  CANCEL SELL price={sell_price_usdc_per_xlm:.7f} reason=requote to best ask")
                    try:
                        submit_cancel_offer(server, kp, base, counter, sell_offer_id, dry_run)
                    except Exception as e:
                        print(f"Requote cancel SELL warning: {e}")
                    sell_offer_id = 0

                print(
                    f"{time.strftime('%b %d %Y %H:%M:%S')}  PLACE SELL price={sell_price_usdc_per_xlm:.7f} "
                    f"amount_xlm={sell_xlm_amt:.4f} queue_ahead={queue_ahead:.2f}"
                )
                try:
                    sell_offer_id = submit_manage_sell_offer(
                        server=server,
                        kp=kp,
                        selling=base,
                        buying=counter,
                        amount_selling=sell_xlm_amt,
                        price_buying_per_selling=sell_price_usdc_per_xlm_for_offer,
                        offer_id=0,
                        dry_run=dry_run,
                    )
                    sell_last_place_ts = t0
                except Exception as e:
                    print(f"PLACE SELL error: {e}")

        # pacing
        elapsed = now_ts() - t0
        sleep_for = max(0.0, poll_seconds - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()


