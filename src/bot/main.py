import os
import time
from decimal import Decimal

import ccxt
from dotenv import load_dotenv

load_dotenv()

D = Decimal


def d(x) -> Decimal:
    return D(str(x))


EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
SYMBOL = os.getenv("SYMBOL", "XRP/USD")

POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))

# Set realistic starter thresholds; tune later
MIN_SPREAD_PCT = d(os.getenv("MIN_SPREAD_PCT", "0.20"))  # 0.20%
MAKER_FEE_PCT = d(os.getenv("MAKER_FEE_PCT", "0.02"))     # 0.02%
BUFFER_PCT = d(os.getenv("BUFFER_PCT", "0.05"))           # 0.05%


def make_exchange():
    ex_class = getattr(ccxt, EXCHANGE_ID)
    ex = ex_class({
        "apiKey": os.getenv("API_KEY"),
        "secret": os.getenv("API_SECRET"),
        "enableRateLimit": True,
    })
    return ex


def best_bid_ask(order_book):
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    if not bids or not asks:
        raise RuntimeError("Empty order book")
    bid = d(bids[0][0])
    ask = d(asks[0][0])
    return bid, ask


def spread_pct(bid: Decimal, ask: Decimal) -> Decimal:
    return (ask - bid) / ask * d("100")


def required_spread_pct() -> Decimal:
    # buy maker + sell maker + safety buffer
    return MAKER_FEE_PCT + MAKER_FEE_PCT + BUFFER_PCT


def main():
    ex = make_exchange()
    ex.load_markets()

    req = required_spread_pct()
    trigger = max(req, MIN_SPREAD_PCT)

    print(f"Exchange: {EXCHANGE_ID} | Symbol: {SYMBOL}")
    print(f"Required spread >= {req}% | Trigger >= {trigger}%")
    print("Mode: PAPER (no orders are placed)\n")

    while True:
        try:
            ob = ex.fetch_order_book(SYMBOL, limit=10)
            bid, ask = best_bid_ask(ob)
            sp = spread_pct(bid, ask)

            print(f"bid={bid} ask={ask} spread={sp:.4f}%")

            if sp >= trigger:
                print(f"  OPPORTUNITY: spread {sp:.4f}% >= {trigger}% -> WOULD try maker BUY then maker SELL\n")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
