import os
import time
import math
import requests
from decimal import Decimal, getcontext

getcontext().prec = 28

# -----------------------------
# Env helpers
# -----------------------------

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else str(v)

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def D(x) -> Decimal:
    return Decimal(str(x))

# -----------------------------
# Config (variables)
# -----------------------------

HORIZON_URL = env_str("HORIZON_URL", "https://horizon.stellar.org")

# Pair
BASE_CODE = env_str("BASE_CODE", "XLM")      # native
BASE_ISSUER = env_str("BASE_ISSUER", "")     # empty for native

QUOTE_CODE = env_str("QUOTE_CODE", "USDC")
QUOTE_ISSUER = env_str("QUOTE_ISSUER", "")   # required for USDC on Stellar

# Trigger range in percent (e.g. 0.08 means 0.08%)
MIN_TRIGGER_PCT = env_float("MIN_TRIGGER_PCT", 0.08)
MAX_TRIGGER_PCT = env_float("MAX_TRIGGER_PCT", 0.13)

# Loop cadence
POLL_SEC = env_float("POLL_SEC", 1.0)
SUMMARY_SEC = env_float("SUMMARY_SEC", 300.0)

# Paper-trade sizing & behavior
TRADE_USDC = env_float("TRADE_USDC", 10.0)  # how much USDC per "trade"
HOLD_SECONDS = env_float("HOLD_SECONDS", 8.0)
SELL_DELAY_SECONDS = env_float("SELL_DELAY_SECONDS", 3.0)
SELL_TIMEOUT_SECONDS = env_float("SELL_TIMEOUT_SECONDS", 8.0)

# Pricing & cost model
INSIDE_PCT = env_float("INSIDE_PCT", 0.002)  # 0.2% (0.002) recommended
FEE_PCT = env_float("FEE_PCT", 0.02)         # 2% conservative buffer

# Safety: live trading disabled by default
TRADING_ENABLED = env_bool("TRADING_ENABLED", False)


# -----------------------------
# Stellar/Horizon asset params
# Horizon uses:
#   native: asset_type=native
#   credit: asset_type=credit_alphanum4 / credit_alphanum12 + asset_code + asset_issuer
# For order_book/trades endpoints:
#   selling_asset_* and buying_asset_* (or base/counter on some SDKs)
# We'll use the canonical Horizon parameter names:
#   selling_asset_type, selling_asset_code, selling_asset_issuer
#   buying_asset_type, buying_asset_code, buying_asset_issuer
# -----------------------------

def asset_type_from_code(code: str) -> str:
    # XLM is native
    if code.upper() == "XLM":
        return "native"
    # Horizon needs alphanum4 or alphanum12
    L = len(code)
    if 1 <= L <= 4:
        return "credit_alphanum4"
    if 5 <= L <= 12:
        return "credit_alphanum12"
    raise ValueError(f"Invalid asset code length for {code!r}")

def add_asset_params(params: dict, prefix: str, code: str, issuer: str):
    """
    prefix is either 'selling' or 'buying'
    """
    code = (code or "").strip()
    issuer = (issuer or "").strip()

    at = asset_type_from_code(code)
    params[f"{prefix}_asset_type"] = at

    if at != "native":
        if not code:
            raise ValueError(f"{prefix}: code is required for non-native asset")
        if not issuer:
            raise ValueError(f"{prefix}: issuer is required for non-native asset")
        params[f"{prefix}_asset_code"] = code
        params[f"{prefix}_asset_issuer"] = issuer

def pair_label() -> str:
    return f"{BASE_CODE.upper()}/{QUOTE_CODE.upper()}"

# We interpret "BASE/QUOTE" as:
# - selling = BASE (XLM)
# - buying  = QUOTE (USDC)
# This matches the idea: use order_book to view XLM/USDC price.
def build_orderbook_params() -> dict:
    params = {}
    add_asset_params(params, "selling", BASE_CODE, BASE_ISSUER)
    add_asset_params(params, "buying", QUOTE_CODE, QUOTE_ISSUER)
    return params

def build_trades_params(order="desc", limit=1) -> dict:
    params = {}
    add_asset_params(params, "selling", BASE_CODE, BASE_ISSUER)
    add_asset_params(params, "buying", QUOTE_CODE, QUOTE_ISSUER)
    params["order"] = order
    params["limit"] = int(limit)
    return params


# -----------------------------
# Horizon fetchers
# -----------------------------

def fetch_order_book() -> dict:
    """
    Horizon /order_book gives:
      bids: [{price, amount, ...}]
      asks: [{price, amount, ...}]
    price here is "price of selling asset in terms of buying asset"
    For selling=XLM, buying=USDC => price is USDC per XLM
    """
    url = f"{HORIZON_URL.rstrip('/')}/order_book"
    params = build_orderbook_params()
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_latest_trade_token() -> str | None:
    """
    Gets the latest trade and returns paging_token (cursor).
    Using requests params prevents malformed URLs (your crash).
    """
    url = f"{HORIZON_URL.rstrip('/')}/trades"
    params = build_trades_params(order="desc", limit=1)

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        recs = data.get("_embedded", {}).get("records", [])
        if not recs:
            return None
        return recs[0].get("paging_token")
    except requests.RequestException as e:
        print(f"Error fetching latest trade token: {e}")
        return None


# -----------------------------
# Math helpers
# -----------------------------

def spread_pct(best_bid: Decimal, best_ask: Decimal) -> Decimal:
    if best_ask <= 0:
        return D(0)
    return (best_ask - best_bid) / best_ask * D(100)

def in_trigger_range(sp_pct: Decimal) -> bool:
    return D(MIN_TRIGGER_PCT) <= sp_pct <= D(MAX_TRIGGER_PCT)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# -----------------------------
# PAPER trading simulation
# -----------------------------

class PaperStats:
    def __init__(self):
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl_usdc = D(0)
        self.gross_usdc = D(0)
        self.fees_usdc = D(0)

    def record(self, pnl: Decimal, gross: Decimal, fees: Decimal):
        self.trades += 1
        self.pnl_usdc += pnl
        self.gross_usdc += gross
        self.fees_usdc += fees
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

    def summary_line(self) -> str:
        avg = (self.pnl_usdc / D(self.trades)) if self.trades else D(0)
        return (
            f"[SUMMARY] trades={self.trades} wins={self.wins} losses={self.losses} "
            f"pnl={self.pnl_usdc:.6f} USDC (avg {avg:.6f}) "
            f"gross={self.gross_usdc:.6f} fees={self.fees_usdc:.6f}"
        )


def paper_buy_sell(best_bid: Decimal, best_ask: Decimal, stats: PaperStats):
    """
    Option A (your request):
      if spread triggers -> BUY then SELL after a few seconds
    In paper-mode we simulate:
      buy_price  ~ best_ask adjusted inside
      sell_price ~ best_bid adjusted inside
    """
    inside = D(INSIDE_PCT) * D(100)  # convert 0.002 -> 0.2% (as percent)
    fee_pct = D(FEE_PCT)

    # Buy slightly better than ask (inside)
    buy_price = best_ask * (D(1) - D(INSIDE_PCT))
    # Sell slightly better than bid (inside)
    sell_price = best_bid * (D(1) + D(INSIDE_PCT))

    # Use TRADE_USDC as quote size
    quote_in = D(TRADE_USDC)
    if buy_price <= 0:
        return

    base_bought = quote_in / buy_price  # XLM amount in paper

    # Wait / hold (simulate time exposure)
    time.sleep(float(HOLD_SECONDS))
    time.sleep(float(SELL_DELAY_SECONDS))

    # Simulate sell proceeds
    quote_out = base_bought * sell_price

    # Conservative fee model: apply fee_pct to both legs
    fees = (quote_in * fee_pct) + (quote_out * fee_pct)
    gross = quote_out - quote_in
    pnl = gross - fees

    stats.record(pnl=pnl, gross=gross, fees=fees)

    print(
        f"[PAPER TRADE] buy@{buy_price:.7f} sell@{sell_price:.7f} "
        f"in={quote_in:.4f} out={quote_out:.4f} gross={gross:.6f} "
        f"fees={fees:.6f} pnl={pnl:.6f}"
    )


# -----------------------------
# Main loop
# -----------------------------

def validate_config():
    # USDC requires issuer
    if QUOTE_CODE.upper() != "XLM" and not QUOTE_ISSUER:
        raise ValueError("QUOTE_ISSUER is required for non-native QUOTE asset (e.g., USDC).")
    # If BASE is not XLM, issuer required too
    if BASE_CODE.upper() != "XLM" and not BASE_ISSUER:
        raise ValueError("BASE_ISSUER is required for non-native BASE asset.")

    if MIN_TRIGGER_PCT <= 0 or MAX_TRIGGER_PCT <= 0:
        raise ValueError("Trigger % must be > 0")
    if MIN_TRIGGER_PCT > MAX_TRIGGER_PCT:
        raise ValueError("MIN_TRIGGER_PCT must be <= MAX_TRIGGER_PCT")


def main():
    validate_config()

    print("Starting Container")
    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {pair_label()}")
    print(f"Trigger range: {MIN_TRIGGER_PCT:.2f}% .. {MAX_TRIGGER_PCT:.2f}%")
    print("Mode:", "LIVE" if TRADING_ENABLED else "PAPER", "(no offers are placed)" if not TRADING_ENABLED else "")
    print(
        f"Paper config: TRADE_USDC={TRADE_USDC}, INSIDE_PCT={INSIDE_PCT*100:.3f}%, "
        f"HOLD_SECONDS={HOLD_SECONDS}, SELL_DELAY_SECONDS={SELL_DELAY_SECONDS}, "
        f"SELL_TIMEOUT_SECONDS={SELL_TIMEOUT_SECONDS}, FEE_PCT={FEE_PCT*100:.2f}%"
    )
    print("")

    # Not required, but useful to ensure trades endpoint is configured correctly
    cursor = fetch_latest_trade_token()
    if cursor is None:
        print("Note: Could not fetch latest trade cursor (Horizon might be rate-limiting). Continuing...\n")
    else:
        print(f"Latest trade cursor: {cursor}\n")

    stats = PaperStats()
    last_summary = time.time()

    while True:
        try:
            ob = fetch_order_book()
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if not bids or not asks:
                print("Empty order book (no bids/asks).")
                time.sleep(POLL_SEC)
                continue

            best_bid = D(bids[0]["price"])
            best_ask = D(asks[0]["price"])
            sp = spread_pct(best_bid, best_ask)

            print(f"bid={best_bid:.7f} ask={best_ask:.7f} spread={sp:.4f}%")

            if in_trigger_range(sp):
                print(
                    f"OPPORTUNITY: spread {sp:.4f}% within "
                    f"{MIN_TRIGGER_PCT:.2f}%..{MAX_TRIGGER_PCT:.2f}%"
                )

                if TRADING_ENABLED:
                    # Live trading is not implemented in this file (by design safety).
                    # When you’re ready, we can add real offer management + signing.
                    print("[LIVE] TRADING_ENABLED=true but live trading is not implemented yet in this version.")
                else:
                    paper_buy_sell(best_bid, best_ask, stats)

            # Periodic summary
            now = time.time()
            if now - last_summary >= SUMMARY_SEC:
                print(stats.summary_line())
                last_summary = now

        except requests.HTTPError as e:
            # Print Horizon error body for debugging, but do not crash
            try:
                resp = e.response.json()
                print("Horizon error response:", resp)
            except Exception:
                pass
            print(f"HTTP error: {e}")
            time.sleep(POLL_SEC)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
