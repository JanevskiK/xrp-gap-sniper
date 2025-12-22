TRADING_ENABLED = False

import os
import time
from decimal import Decimal, InvalidOperation

import requests
from dotenv import load_dotenv

load_dotenv()
D = Decimal

HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").strip()
POLL_SEC = float(os.getenv("POLL_SEC", "1.0").strip())

MIN_TRIGGER_PCT = D(os.getenv("MIN_TRIGGER_PCT", "0.08").strip())
MAX_TRIGGER_PCT = D(os.getenv("MAX_TRIGGER_PCT", "0.13").strip())

BASE_CODE = os.getenv("BASE_CODE", "XLM").strip()
BASE_ISSUER = os.getenv("BASE_ISSUER", "").strip()

QUOTE_CODE = os.getenv("QUOTE_CODE", "USDC").strip()
QUOTE_ISSUER = os.getenv("QUOTE_ISSUER", "").strip()

TRADING_ENABLED = False  # PAPER only


def spread_pct(best_bid: Decimal, best_ask: Decimal) -> Decimal:
    return (best_ask - best_bid) / best_ask * D("100")


def in_trigger_range(x: Decimal) -> bool:
    return MIN_TRIGGER_PCT <= x <= MAX_TRIGGER_PCT


def asset_params(prefix: str, code: str, issuer: str) -> dict:
    # Horizon expects base_asset_type/base_asset_code/base_asset_issuer, etc.
    if code.upper() == "XLM" and not issuer:
        return {f"{prefix}_asset_type": "native"}

    if not issuer:
        raise ValueError(f"{code} requires an issuer. Set {prefix.upper()}_ISSUER.")

    # Most common for USDC is credit_alphanum4
    return {
        f"{prefix}_asset_type": "credit_alphanum4" if len(code) <= 4 else "credit_alphanum12",
        f"{prefix}_asset_code": code,
        f"{prefix}_asset_issuer": issuer,
    }


def fetch_order_book() -> dict:
    params = {}
    params.update(asset_params("base", BASE_CODE, BASE_ISSUER))
    params.update(asset_params("counter", QUOTE_CODE, QUOTE_ISSUER))  # Horizon uses "counter_*"

    url = f"{HORIZON_URL.rstrip('/')}/order_book"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def main():
    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {BASE_CODE}/{QUOTE_CODE}")
    print(f"Trigger range: {MIN_TRIGGER_PCT:.2f}% .. {MAX_TRIGGER_PCT:.2f}%")
    print("Mode: PAPER (no offers are placed)\n")

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
