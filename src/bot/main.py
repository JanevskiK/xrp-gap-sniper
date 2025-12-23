import os
import time
from decimal import Decimal, getcontext
from typing import Dict, Tuple, Optional

import requests
from dotenv import load_dotenv

# Only needed for LIVE trading
from stellar_sdk import (
    Keypair,
    Network,
    Server,
    TransactionBuilder,
    Asset,
)
from stellar_sdk.exceptions import BadRequestError, NotFoundError

getcontext().prec = 28
D = Decimal


# -----------------------------
# ENV / CONFIG
# -----------------------------
load_dotenv()

HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").strip()

BASE_CODE = os.getenv("BASE_CODE", "XLM").strip()
BASE_ISSUER = os.getenv("BASE_ISSUER", "").strip()  # MUST be empty for XLM

QUOTE_CODE = os.getenv("QUOTE_CODE", "USDC").strip()
QUOTE_ISSUER = os.getenv("QUOTE_ISSUER", "").strip()

MIN_TRIGGER_PCT = D(os.getenv("MIN_TRIGGER_PCT", "0.08"))  # percent
MAX_TRIGGER_PCT = D(os.getenv("MAX_TRIGGER_PCT", "0.13"))  # percent

POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))

# Option A parameters
TRADE_USDC = D(os.getenv("TRADE_USDC", "10"))  # how much USDC to use per cycle
HOLD_SECONDS = int(os.getenv("HOLD_SECONDS", "8"))  # how long to wait between buy and sell
FEE_PCT = D(os.getenv("FEE_PCT", "0.02"))  # percent safety margin (slippage/fees buffer)

TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false").lower() in ("1", "true", "yes", "y")

# For LIVE trading
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
PUBLIC_KEY = os.getenv("PUBLIC_KEY", "").strip()  # optional; derived from secret if not set

SUMMARY_SEC = int(os.getenv("SUMMARY_SEC", "300"))  # periodic summary prints


# -----------------------------
# ASSET HELPERS
# -----------------------------
def horizon_asset_params(prefix: str, code: str, issuer: str) -> Dict[str, str]:
    """
    Build Horizon params for /order_book which uses selling_* and buying_*.

    For native XLM:
      {prefix}_asset_type=native
    For credit assets:
      {prefix}_asset_type=credit_alphanum4 or credit_alphanum12
      {prefix}_asset_code=...
      {prefix}_asset_issuer=...
    """
    if code.upper() == "XLM" and issuer == "":
        return {f"{prefix}_asset_type": "native"}

    if not issuer:
        raise ValueError(f"{prefix} issuer is empty for non-native asset code={code}")

    asset_type = "credit_alphanum4" if len(code) <= 4 else "credit_alphanum12"
    return {
        f"{prefix}_asset_type": asset_type,
        f"{prefix}_asset_code": code,
        f"{prefix}_asset_issuer": issuer,
    }


def sdk_asset(code: str, issuer: str) -> Asset:
    if code.upper() == "XLM" and issuer == "":
        return Asset.native()
    return Asset(code, issuer)


# -----------------------------
# MARKET DATA
# -----------------------------
def fetch_order_book() -> Dict:
    """
    GET /order_book?selling_asset_type=...&buying_asset_type=...
    We'll treat BASE as "selling" and QUOTE as "buying" so price is QUOTE per BASE.
    For XLM/USDC => price is USDC per XLM.
    """
    params = {}
    params.update(horizon_asset_params("selling", BASE_CODE, BASE_ISSUER))
    params.update(horizon_asset_params("buying", QUOTE_CODE, QUOTE_ISSUER))

    url = f"{HORIZON_URL.rstrip('/')}/order_book"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def best_bid_ask(order_book: Dict) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if not bids or not asks:
        return None, None

    # Horizon returns price as string
    bid = D(bids[0]["price"])  # highest bid
    ask = D(asks[0]["price"])  # lowest ask
    return bid, ask


def spread_pct(bid: Decimal, ask: Decimal) -> Decimal:
    # percent
    if ask <= 0:
        return D("0")
    return (ask - bid) / ask * D("100")


def in_trigger_range(sp: Decimal) -> bool:
    return MIN_TRIGGER_PCT <= sp <= MAX_TRIGGER_PCT


# -----------------------------
# LIVE TRADING (Option A)
# -----------------------------
def load_trading_keys() -> Tuple[str, str]:
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY is required when TRADING_ENABLED=true")

    kp = Keypair.from_secret(SECRET_KEY)
    pub = PUBLIC_KEY or kp.public_key
    return kp.secret, pub

def main():
    print("DEBUG: bot started")
    print(f"DEBUG: BASE={BASE_CODE} issuer={BASE_ISSUER}")
    print(f"DEBUG: QUOTE={QUOTE_CODE} issuer={QUOTE_ISSUER}")


def submit_path_payment_strict_send(
    server: Server,
    source_secret: str,
    source_public: str,
    send_asset: Asset,
    send_amount: Decimal,
    dest_asset: Asset,
    dest_min: Decimal,
    memo_text: str = "gap-bot",
) -> str:
    """
    A "market-style" swap:
    - spend exactly send_amount of send_asset
    - receive at least dest_min of dest_asset
    """
    account = server.load_account(source_public)

    tx = (
        TransactionBuilder(
            source_account=account,
            network_passphrase=Network.PUBLIC_NETWORK_PASSPHRASE,
            base_fee=200,  # slightly above minimum
        )
        .add_text_memo(memo_text[:28])
        .append_path_payment_strict_send_op(
            destination=source_public,  # self
            send_asset=send_asset,
            send_amount=str(send_amount),
            dest_asset=dest_asset,
            dest_min=str(dest_min),
            path=[],
        )
        .set_timeout(60)
        .build()
    )

    tx.sign(Keypair.from_secret(source_secret))
    resp = server.submit_transaction(tx)
    return resp["hash"]


def estimate_buy_xlm_amount(usdc_amount: Decimal, ask: Decimal, safety_fee_pct: Decimal) -> Decimal:
    """
    If ask = USDC per XLM, then expected XLM = USDC / ask
    Apply a safety haircut based on FEE_PCT.
    """
    if ask <= 0:
        return D("0")
    expected = usdc_amount / ask
    haircut = D("1") - (safety_fee_pct / D("100"))
    return (expected * haircut).quantize(D("0.0000001"))  # 7 decimals


def estimate_sell_usdc_min(xlm_amount: Decimal, bid: Decimal, safety_fee_pct: Decimal) -> Decimal:
    """
    If bid = USDC per XLM, expected USDC = XLM * bid
    Apply safety haircut.
    """
    expected = xlm_amount * bid
    haircut = D("1") - (safety_fee_pct / D("100"))
    return (expected * haircut).quantize(D("0.0000001"))


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    print("Starting Container")
    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {BASE_CODE}/{QUOTE_CODE}")
    print(f"Trigger range: {MIN_TRIGGER_PCT:.2f}% .. {MAX_TRIGGER_PCT:.2f}%")
    print(f"Mode: {'LIVE' if TRADING_ENABLED else 'PAPER'}")
    print(
        f"Config: TRADE_USDC={TRADE_USDC}, HOLD_SECONDS={HOLD_SECONDS}, "
        f"POLL_SEC={POLL_SEC}, FEE_PCT={FEE_PCT}%"
    )
    print("")

    server = None
    source_secret = None
    source_public = None
    if TRADING_ENABLED:
        source_secret, source_public = load_trading_keys()
        server = Server(HORIZON_URL)
        print(f"LIVE wallet: {source_public}")
        print("")

    last_summary = time.time()
    opportunities = 0

    while True:
        try:
            ob = fetch_order_book()
            bid, ask = best_bid_ask(ob)
            if bid is None or ask is None:
                print("Empty order book (no bids/asks).")
                time.sleep(POLL_SEC)
                continue

            sp = spread_pct(bid, ask)
            print(f"bid={bid} ask={ask} spread={sp:.4f}%")

            if in_trigger_range(sp):
                opportunities += 1
                print(f"OPPORTUNITY: spread {sp:.4f}% is within {MIN_TRIGGER_PCT}%..{MAX_TRIGGER_PCT}%")

                if not TRADING_ENABLED:
                    print("PAPER: would BUY XLM with USDC, wait, then SELL XLM back to USDC.\n")
                else:
                    # BUY: spend USDC, receive XLM (min based on ask)
                    usdc_asset = sdk_asset(QUOTE_CODE, QUOTE_ISSUER)
                    xlm_asset = sdk_asset(BASE_CODE, BASE_ISSUER)

                    buy_min_xlm = estimate_buy_xlm_amount(TRADE_USDC, ask, FEE_PCT)
                    if buy_min_xlm <= 0:
                        print("LIVE: buy_min_xlm computed as 0; skipping.\n")
                        time.sleep(POLL_SEC)
                        continue

                    print(f"LIVE: BUY spending {TRADE_USDC} {QUOTE_CODE} for at least {buy_min_xlm} {BASE_CODE}")
                    buy_hash = submit_path_payment_strict_send(
                        server=server,
                        source_secret=source_secret,
                        source_public=source_public,
                        send_asset=usdc_asset,
                        send_amount=TRADE_USDC,
                        dest_asset=xlm_asset,
                        dest_min=buy_min_xlm,
                        memo_text="gap-buy",
                    )
                    print(f"LIVE: BUY tx hash: {buy_hash}")

                    print(f"LIVE: waiting {HOLD_SECONDS}s before SELL ...")
                    time.sleep(HOLD_SECONDS)

                    # SELL: spend XLM, receive USDC (min based on bid)
                    # We sell the amount we expected to get at minimum from the buy.
                    sell_min_usdc = estimate_sell_usdc_min(buy_min_xlm, bid, FEE_PCT)
                    if sell_min_usdc <= 0:
                        print("LIVE: sell_min_usdc computed as 0; skipping sell.\n")
                        time.sleep(POLL_SEC)
                        continue

                    print(f"LIVE: SELL spending {buy_min_xlm} {BASE_CODE} for at least {sell_min_usdc} {QUOTE_CODE}")
                    sell_hash = submit_path_payment_strict_send(
                        server=server,
                        source_secret=source_secret,
                        source_public=source_public,
                        send_asset=xlm_asset,
                        send_amount=buy_min_xlm,
                        dest_asset=usdc_asset,
                        dest_min=sell_min_usdc,
                        memo_text="gap-sell",
                    )
                    print(f"LIVE: SELL tx hash: {sell_hash}\n")

            # periodic summary
            now = time.time()
            if now - last_summary >= SUMMARY_SEC:
                print(f"\nSummary: opportunities seen in last {SUMMARY_SEC}s: {opportunities}\n")
                opportunities = 0
                last_summary = now

            time.sleep(POLL_SEC)

        except requests.HTTPError as e:
            # Horizon returned a non-2xx
            try:
                detail = e.response.json()
                print(f"Horizon HTTPError: {detail}")
            except Exception:
                print(f"Horizon HTTPError: {e}")
            time.sleep(POLL_SEC)

        except (BadRequestError, NotFoundError) as e:
            print(f"Stellar SDK error: {e}")
            time.sleep(POLL_SEC)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
