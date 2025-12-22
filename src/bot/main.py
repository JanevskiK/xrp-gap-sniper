import os
import time
import requests
from decimal import Decimal, getcontext
from datetime import datetime

getcontext().prec = 28
D = Decimal


# ----------------------------
# ENV (Railway Variables)
# ----------------------------
HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").rstrip("/")

BASE_CODE = os.getenv("BASE_CODE", "XLM")
BASE_ISSUER = os.getenv("BASE_ISSUER", "")  # empty for native XLM

QUOTE_CODE = os.getenv("QUOTE_CODE", "USDC")
QUOTE_ISSUER = os.getenv("QUOTE_ISSUER", "")  # required for USDC

MIN_TRIGGER_PCT = D(os.getenv("MIN_TRIGGER_PCT", "0.08"))  # percent
MAX_TRIGGER_PCT = D(os.getenv("MAX_TRIGGER_PCT", "0.13"))  # percent

POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))

# Paper trading config
TRADE_USDC = D(os.getenv("TRADE_USDC", "10"))          # paper size in USDC
HOLD_SECONDS = int(os.getenv("HOLD_SECONDS", "8"))     # max time to wait for buy fill
SELL_DELAY_SECONDS = int(os.getenv("SELL_DELAY_SECONDS", "3"))  # after buy fill, wait then try sell
SELL_TIMEOUT_SECONDS = int(os.getenv("SELL_TIMEOUT_SECONDS", "8"))

# How far inside the spread to quote (percent)
# 0.01 means 0.01% inside
INSIDE_PCT = D(os.getenv("INSIDE_PCT", "0.01"))        # percent

# Estimated total fee/slippage buffer (percent of notional)
# Stellar base fees are tiny, but adverse selection/slippage isn't.
FEE_PCT = D(os.getenv("FEE_PCT", "0.02"))              # percent

SUMMARY_SEC = int(os.getenv("SUMMARY_SEC", "300"))     # stats print interval


# ----------------------------
# Helpers: assets + requests
# ----------------------------
def is_native(code: str, issuer: str) -> bool:
    return code.upper() == "XLM" and (issuer or "") == ""


def orderbook_params_for_asset(prefix: str, code: str, issuer: str) -> dict:
    """
    /order_book expects:
      selling_asset_type, selling_asset_code, selling_asset_issuer
      buying_asset_type,  buying_asset_code,  buying_asset_issuer
    """
    if is_native(code, issuer):
        return {f"{prefix}_asset_type": "native"}
    else:
        # USDC is credit_alphanum4
        asset_type = "credit_alphanum4" if len(code) <= 4 else "credit_alphanum12"
        return {
            f"{prefix}_asset_type": asset_type,
            f"{prefix}_asset_code": code,
            f"{prefix}_asset_issuer": issuer,
        }


def trades_params_for_pair(base_code: str, base_issuer: str, counter_code: str, counter_issuer: str) -> dict:
    """
    /trades uses base_asset_* and counter_asset_* (this endpoint still uses base/counter naming).
    """
    params = {}

    # base
    if is_native(base_code, base_issuer):
        params["base_asset_type"] = "native"
    else:
        params["base_asset_type"] = "credit_alphanum4" if len(base_code) <= 4 else "credit_alphanum12"
        params["base_asset_code"] = base_code
        params["base_asset_issuer"] = base_issuer

    # counter
    if is_native(counter_code, counter_issuer):
        params["counter_asset_type"] = "native"
    else:
        params["counter_asset_type"] = "credit_alphanum4" if len(counter_code) <= 4 else "credit_alphanum12"
        params["counter_asset_code"] = counter_code
        params["counter_asset_issuer"] = counter_issuer

    return params


def fetch_order_book() -> dict:
    # We want a book that effectively returns USDC-per-XLM prices like you were seeing.
    # Use: selling = BASE (XLM), buying = QUOTE (USDC)
    params = {}
    params.update(orderbook_params_for_asset("selling", BASE_CODE, BASE_ISSUER))
    params.update(orderbook_params_for_asset("buying", QUOTE_CODE, QUOTE_ISSUER))

    url = f"{HORIZON_URL}/order_book"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_latest_trade_token() -> str | None:
    """
    Get the most recent trade paging token so we can stream forward from "now".
    """
    url = f"{HORIZON_URL}/trades"
    params = trades_params_for_pair(BASE_CODE, BASE_ISSUER, QUOTE_CODE, QUOTE_ISSUER)
    params.update({"order": "desc", "limit": 1})
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    records = r.json().get("_embedded", {}).get("records", [])
    if not records:
        return None
    return records[0].get("paging_token")


def fetch_trades_since(cursor: str | None, limit: int = 200) -> tuple[list[dict], str | None]:
    """
    Fetch trades after 'cursor' (ascending). Returns (records, last_token).
    """
    url = f"{HORIZON_URL}/trades"
    params = trades_params_for_pair(BASE_CODE, BASE_ISSUER, QUOTE_CODE, QUOTE_ISSUER)
    params.update({"order": "asc", "limit": limit})
    if cursor:
        params["cursor"] = cursor

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    records = r.json().get("_embedded", {}).get("records", [])
    last_token = records[-1]["paging_token"] if records else cursor
    return records, last_token


def spread_pct(best_bid: Decimal, best_ask: Decimal) -> Decimal:
    if best_bid <= 0:
        return D("0")
    return ((best_ask - best_bid) / best_bid) * D("100")


def in_trigger(sp: Decimal) -> bool:
    return MIN_TRIGGER_PCT <= sp <= MAX_TRIGGER_PCT


def inside_price(price: Decimal, pct: Decimal, direction: str) -> Decimal:
    """
    direction:
      'up'   -> price * (1 + pct/100)
      'down' -> price * (1 - pct/100)
    """
    p = pct / D("100")
    if direction == "up":
        return price * (D("1") + p)
    return price * (D("1") - p)


def safe_dec(x) -> Decimal:
    return D(str(x))


# ----------------------------
# Paper simulation
# ----------------------------
class Stats:
    def __init__(self):
        self.opportunities = 0
        self.buy_attempts = 0
        self.buy_fills = 0
        self.sell_attempts = 0
        self.sell_fills = 0
        self.pnl_usdc = D("0")
        self.last_summary = time.time()

    def summary(self):
        return (
            f"\n--- PAPER STATS ---\n"
            f"opportunities: {self.opportunities}\n"
            f"buy attempts:  {self.buy_attempts}\n"
            f"buy fills:     {self.buy_fills}\n"
            f"sell attempts: {self.sell_attempts}\n"
            f"sell fills:    {self.sell_fills}\n"
            f"est PnL USDC:  {self.pnl_usdc:.6f}\n"
            f"--------------\n"
        )


def trade_price_usdc_per_xlm(tr: dict) -> Decimal | None:
    """
    Horizon trade record includes 'base_is_seller' and 'price' object.
    We avoid overthinking: many Horizon trade records include a 'price' object with 'n'/'d',
    and also 'base_amount'/'counter_amount'. We'll compute counter/base as USDC per XLM.
    """
    try:
        base_amt = safe_dec(tr.get("base_amount"))
        counter_amt = safe_dec(tr.get("counter_amount"))
        if base_amt == 0:
            return None
        return counter_amt / base_amt
    except Exception:
        return None


def would_fill_buy(trades: list[dict], buy_price: Decimal) -> bool:
    # We get filled buying XLM if trade prints at price <= our bid.
    for tr in trades:
        p = trade_price_usdc_per_xlm(tr)
        if p is not None and p <= buy_price:
            return True
    return False


def would_fill_sell(trades: list[dict], sell_price: Decimal) -> bool:
    # We get filled selling XLM if trade prints at price >= our ask.
    for tr in trades:
        p = trade_price_usdc_per_xlm(tr)
        if p is not None and p >= sell_price:
            return True
    return False


def main():
    print("Starting Container", flush=True)
    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {BASE_CODE}/{QUOTE_CODE}")
    print(f"Trigger range: {MIN_TRIGGER_PCT:.2f}% .. {MAX_TRIGGER_PCT:.2f}%")
    print("Mode: PAPER (no offers are placed)")
    print(
        f"Paper config: TRADE_USDC={TRADE_USDC}, INSIDE_PCT={INSIDE_PCT}%, "
        f"HOLD_SECONDS={HOLD_SECONDS}, SELL_DELAY_SECONDS={SELL_DELAY_SECONDS}, SELL_TIMEOUT_SECONDS={SELL_TIMEOUT_SECONDS}, "
        f"FEE_PCT={FEE_PCT}%\n"
    )

    stats = Stats()

    # Start trade cursor at "now" so we only watch future trades
    trade_cursor = fetch_latest_trade_token()

    while True:
        try:
            ob = fetch_order_book()
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if not bids or not asks:
                print("Empty order book (no bids/asks).")
                time.sleep(POLL_SEC)
                continue

            best_bid = safe_dec(bids[0]["price"])
            best_ask = safe_dec(asks[0]["price"])
            sp = spread_pct(best_bid, best_ask)

            print(f"bid={best_bid} ask={best_ask} spread={sp:.4f}%")

            # print periodic stats
            now = time.time()
            if now - stats.last_summary >= SUMMARY_SEC:
                print(stats.summary())
                stats.last_summary = now

            if not in_trigger(sp):
                time.sleep(POLL_SEC)
                continue

            stats.opportunities += 1

            # Paper BUY: slightly above best bid (inside spread)
            buy_price = inside_price(best_bid, INSIDE_PCT, "up")
            stats.buy_attempts += 1
            print(f"🎯 TRIGGER: spread in range. PAPER BUY @ {buy_price} (inside +{INSIDE_PCT}%)")

            # Wait up to HOLD_SECONDS to see if trades print through us
            buy_filled = False
            t_end = time.time() + HOLD_SECONDS

            while time.time() < t_end:
                new_trades, trade_cursor = fetch_trades_since(trade_cursor, limit=200)
                if new_trades and would_fill_buy(new_trades, buy_price):
                    buy_filled = True
                    break
                time.sleep(1.0)

            if not buy_filled:
                print("⛔ PAPER BUY not filled within timeout. Cancel (paper).")
                time.sleep(POLL_SEC)
                continue

            stats.buy_fills += 1
            base_qty = (TRADE_USDC / buy_price)  # XLM amount acquired (paper)
            print(f"✅ PAPER BUY filled. qty≈{base_qty:.7f} XLM  (spent {TRADE_USDC} USDC)")

            # Wait a bit before selling
            time.sleep(max(0, SELL_DELAY_SECONDS))

            # Refresh orderbook for sell pricing
            ob2 = fetch_order_book()
            bids2 = ob2.get("bids", [])
            asks2 = ob2.get("asks", [])
            if not bids2 or not asks2:
                print("Orderbook empty after buy; skipping sell (paper).")
                time.sleep(POLL_SEC)
                continue

            best_bid2 = safe_dec(bids2[0]["price"])
            best_ask2 = safe_dec(asks2[0]["price"])

            # Paper SELL: slightly below best ask (inside spread)
            sell_price = inside_price(best_ask2, INSIDE_PCT, "down")
            stats.sell_attempts += 1
            print(f"📌 PAPER SELL @ {sell_price} (inside -{INSIDE_PCT}%)")

            sell_filled = False
            t_end2 = time.time() + SELL_TIMEOUT_SECONDS

            while time.time() < t_end2:
                new_trades, trade_cursor = fetch_trades_since(trade_cursor, limit=200)
                if new_trades and would_fill_sell(new_trades, sell_price):
                    sell_filled = True
                    break
                time.sleep(1.0)

            if not sell_filled:
                print("⛔ PAPER SELL not filled within timeout. Cancel (paper).")
                time.sleep(POLL_SEC)
                continue

            stats.sell_fills += 1

            gross_proceeds = base_qty * sell_price  # USDC
            gross_pnl = gross_proceeds - TRADE_USDC

            # Apply a conservative fee/slippage buffer on notional (round-trip)
            # Buffer is applied to the full notional twice (buy+sell) roughly, so 2x.
            buffer = (TRADE_USDC * (FEE_PCT / D("100"))) * D("2")
            net_pnl = gross_pnl - buffer

            stats.pnl_usdc += net_pnl

            print(
                f"✅ PAPER SELL filled.\n"
                f"   gross proceeds: {gross_proceeds:.6f} USDC\n"
                f"   gross PnL:      {gross_pnl:.6f} USDC\n"
                f"   buffer (fees):  {buffer:.6f} USDC\n"
                f"   NET PnL:        {net_pnl:.6f} USDC\n"
            )

            time.sleep(POLL_SEC)

        except requests.HTTPError as e:
            # Print Horizon error body if available
            try:
                print(f"HTTPError: {e} | response: {e.response.text[:600]}")
            except Exception:
                print(f"HTTPError: {e}")
            time.sleep(POLL_SEC)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
