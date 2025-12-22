
import os
import time
from decimal import Decimal

import requests
from dotenv import load_dotenv

load_dotenv()
D = Decimal

# ---- Config (Railway Variables / .env) ----
HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").strip()

BASE_CODE = os.getenv("BASE_CODE", "XLM").strip()
BASE_ISSUER = os.getenv("BASE_ISSUER", "").strip()  # keep blank for XLM (native)

QUOTE_CODE = os.getenv("QUOTE_CODE", "USDC").strip()
QUOTE_ISSUER = os.getenv("QUOTE_ISSUER", "").strip()

MIN_TRIGGER_PCT = D(os.getenv("MIN_TRIGGER_PCT", "0.08").strip())  # %
MAX_TRIGGER_PCT = D(os.getenv("MAX_TRIGGER_PCT", "0.13").strip())  # %

POLL_SEC = float(os.getenv("POLL_SEC", "1.0").strip())

# Paper mode by default (no real trading yet)
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false").strip().lower() in ("1", "true", "yes")


def validate_issuer(code: str, issuer: str, env_name: str) -> None:
    """Issuer should be a Stellar public key: starts with G and length 56."""
    if code.upper() == "XLM":
        return
    if not issuer:
        raise ValueError(f"{env_name} is required for non-native asset {code}.")
    if not (issuer.startswith("G") and len(issuer) == 56):
        raise ValueError(f"{env_name} looks invalid: {issuer!r} (should be 56 chars, starting with 'G').")


def asset_params(prefix: str, code: str, issuer: str) -> dict:
    """
    Build Horizon /order_book params for selling_* or buying_*.

    /order_book expects:
      selling_asset_type / buying_asset_type: native | credit_alphanum4 | credit_alphanum12
      selling_asset_code / buying_asset_code (if not native)
      selling_asset_issuer / buying_asset_issuer (if not native)
    """
    code = code.strip()
    issuer = issuer.strip()

    if code.upper() == "XLM" and issuer == "":
        return {f"{prefix}_asset_type": "native"}

    if not issuer:
        raise ValueError(f"{code} requires an issuer. Set {prefix.upper()}_ISSUER.")

    asset_type = "credit_alphanum4" if len(code) <= 4 else "credit_alphanum12"

    return {
        f"{prefix}_asset_type": asset_type,
        f"{prefix}_asset_code": code,
        f"{prefix}_asset_issuer": issuer,
    }


def fetch_order_book() -> dict:
    # Native XLM must have blank issuer
    selling_issuer = "" if BASE_CODE.upper() == "XLM" else BASE_ISSUER

    validate_issuer(BASE_CODE, selling_issuer, "BASE_ISSUER")
    validate_issuer(QUOTE_CODE, QUOTE_ISSUER, "QUOTE_ISSUER")

    params = {}
    params.update(asset_params("selling", BASE_CODE, selling_issuer))
    params.update(asset_params("buying", QUOTE_CODE, QUOTE_ISSUER))

    url = f"{HORIZON_URL.rstrip('/')}/order_book"
    r = requests.get(url, params=params, timeout=10)

    if r.status_code >= 400:
        # Print Horizon's error message to the logs (super useful for debugging)
        try:
            print("Horizon error response:", r.json())
        except Exception:
            print("Horizon error response (text):", r.text)
        r.raise_for_status()

    return r.json()


def spread_pct(best_bid: Decimal, best_ask: Decimal) -> Decimal:
    # spread% = (ask - bid) / ask * 100
    return (best_ask - best_bid) / best_ask * D("100")


def in_trigger_range(x: Decimal) -> bool:
    return MIN_TRIGGER_PCT <= x <= MAX_TRIGGER_PCT


def main() -> None:
    print("Starting Container")
    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {BASE_CODE}/{QUOTE_CODE}")
    print(f"Trigger range: {MIN_TRIGGER_PCT:.2f}% .. {MAX_TRIGGER_PCT:.2f}%")
    print("Mode: PAPER (no offers are placed)" if not TRADING_ENABLED else "Mode: LIVE (offers may be placed)")
    print()

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

            print(f"bid={best_bid} ask={best_ask} spread={sp:.4f}%")

            if in_trigger_range(sp):
                print(
                    f"  OPPORTUNITY: spread {sp:.4f}% is within "
                    f"{MIN_TRIGGER_PCT:.2f}%..{MAX_TRIGGER_PCT:.2f}%\n"
                )

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
