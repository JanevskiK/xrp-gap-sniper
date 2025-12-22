TRADING_ENABLED = False

import os
import time
from decimal import Decimal
from dotenv import load_dotenv

from stellar_sdk import Server, Asset

load_dotenv()
D = Decimal

# Stellar Horizon
HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org")
POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))

# Trigger range (percent)
MIN_TRIGGER_PCT = D(os.getenv("MIN_TRIGGER_PCT", "0.08"))  # 0.08%
MAX_TRIGGER_PCT = D(os.getenv("MAX_TRIGGER_PCT", "0.13"))  # 0.13%

# Pair: XLM / USDC (USDC issuer required)
BASE_CODE = os.getenv("BASE_CODE", "XLM")
BASE_ISSUER = os.getenv("BASE_ISSUER", "")  # empty => native XLM

QUOTE_CODE = os.getenv("QUOTE_CODE", "USDC")
QUOTE_ISSUER = os.getenv(
    "QUOTE_ISSUER",
    "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
)

TRADING_ENABLED = False  # hard lock (paper only)


def asset_from_env(code: str, issuer: str) -> Asset:
    if code.upper() == "XLM" and not issuer:
        return Asset.native()
    if not issuer:
        raise ValueError(f"Asset {code} requires an issuer (set *_ISSUER env var).")
    return Asset(code, issuer)


def spread_pct(best_bid: Decimal, best_ask: Decimal) -> Decimal:
    # Percent spread relative to ask
    return (best_ask - best_bid) / best_ask * D("100")


def in_trigger_range(x: Decimal) -> bool:
    return MIN_TRIGGER_PCT <= x <= MAX_TRIGGER_PCT


def main():
    server = Server(HORIZON_URL)
    base = asset_from_env(BASE_CODE, BASE_ISSUER)
    quote = asset_from_env(QUOTE_CODE, QUOTE_ISSUER)

    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {BASE_CODE}/{QUOTE_CODE}")
    print(f"Trigger range: {MIN_TRIGGER_PCT:.2f}% .. {MAX_TRIGGER_PCT:.2f}%")
    print("Mode: PAPER (no offers are placed)\n")

    while True:
        try:
            ob = server.orderbook(base=base, quote=quote).call()
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if not bids or not asks:
                print("Empty order book (no bids/asks).")
                time.sleep(POLL_SEC)
                continue

            best_bid = D(bids[0]["price"])
            best_ask = D(asks[0]["price"])
            sp = spread_pct(best_bid, best_ask)

            # Always print the live spread
            print(f"bid={best_bid} ask={best_ask} spread={sp:.4f}%")

            # Trigger only inside the range you requested
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
