import os
import time
import requests

HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").strip()
BASE_CODE = os.getenv("BASE_CODE", "XLM").strip()
BASE_ISSUER = (os.getenv("BASE_ISSUER") or "").strip()   # empty means native
QUOTE_CODE = os.getenv("QUOTE_CODE", "USDC").strip()
QUOTE_ISSUER = (os.getenv("QUOTE_ISSUER") or "").strip()

POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))

# Paper-trade settings
TRADE_USDC = float(os.getenv("TRADE_USDC", "10"))          # how much quote (USDC) per trade
INSIDE_PCT = float(os.getenv("INSIDE_PCT", "0.002"))       # 0.002 = 0.2% trigger threshold (spread)
FEE_PCT = float(os.getenv("FEE_PCT", "0.02"))              # 0.02 = 2% (if you mean 0.02%) set 0.0002
SUMMARY_SEC = int(os.getenv("SUMMARY_SEC", "300"))         # print summary every N seconds

def asset_type(code: str, issuer: str) -> str:
    if code.upper() == "XLM" and issuer == "":
        return "native"
    return "credit_alphanum4" if len(code) <= 4 else "credit_alphanum12"

def add_asset_params(params: dict, side: str, code: str, issuer: str) -> None:
    a_type = asset_type(code, issuer)
    params[f"{side}_asset_type"] = a_type
    if a_type != "native":
        if not issuer:
            raise ValueError(f"{side} issuer is empty for non-native asset code={code}")
        params[f"{side}_asset_code"] = code
        params[f"{side}_asset_issuer"] = issuer

def fetch_order_book() -> dict:
    # For pair BASE/QUOTE:
    # selling = BASE, buying = QUOTE
    params = {}
    add_asset_params(params, "selling", BASE_CODE, BASE_ISSUER)
    add_asset_params(params, "buying", QUOTE_CODE, QUOTE_ISSUER)

    url = f"{HORIZON_URL.rstrip('/')}/order_book"

    # Optional debug (comment these out once stable)
    # print("DEBUG order_book url:", url)
    # print("DEBUG order_book params:", params)

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def main():
    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {BASE_CODE}/{QUOTE_CODE}")
    print(f"Paper config: TRADE_USDC={TRADE_USDC}, INSIDE_PCT={INSIDE_PCT}, FEE_PCT={FEE_PCT}, SUMMARY_SEC={SUMMARY_SEC}")

    trades = 0
    wins = 0
    total_net = 0.0
    last_summary = time.time()

    while True:
        try:
            ob = fetch_order_book()
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if not bids or not asks:
                print("TICK empty book (no bids/asks)")
                time.sleep(POLL_SEC)
                continue

            # Horizon order_book returns price as string
            best_bid = safe_float(bids[0].get("price"))
            best_ask = safe_float(asks[0].get("price"))

            if best_bid is None or best_ask is None or best_ask <= 0:
                print("TICK invalid bid/ask:", best_bid, best_ask)
                time.sleep(POLL_SEC)
                continue

            # Spread percentage (mid-based is another option; this is ask-based)
            spread_pct = (best_ask - best_bid) / best_ask  # e.g. 0.002 = 0.2%

            print(f"TICK bid={best_bid:.7f} ask={best_ask:.7f} spread={spread_pct*100:.4f}% bids={len(bids)} asks={len(asks)}")

            # Trigger condition
            if spread_pct >= INSIDE_PCT:
                print(f"SIGNAL spread {spread_pct*100:.4f}% >= threshold {INSIDE_PCT*100:.4f}% -> PAPER TRADE")

                # PAPER BUY: spend TRADE_USDC at ask => get BASE
                base_bought = TRADE_USDC / best_ask

                # PAPER SELL: sell BASE at bid => get QUOTE back
                usdc_out = base_bought * best_bid

                gross = usdc_out - TRADE_USDC
                fee = TRADE_USDC * FEE_PCT
                net = gross - fee

                trades += 1
                if net > 0:
                    wins += 1
                total_net += net

                print(f"BUY  {QUOTE_CODE}={TRADE_USDC:.2f} price(ask)={best_ask:.7f} -> {BASE_CODE}={base_bought:.7f}")
                print(f"SELL {BASE_CODE}={base_bought:.7f} price(bid)={best_bid:.7f} -> {QUOTE_CODE}_out={usdc_out:.6f}")
                print(f"P&L gross={gross:.6f} fee={fee:.6f} net={net:.6f} total_net={total_net:.6f} trades={trades} winrate={(wins/trades*100):.1f}%")

            # Periodic summary
            now = time.time()
            if now - last_summary >= SUMMARY_SEC:
                winrate = (wins / trades * 100) if trades else 0.0
                print(f"SUMMARY trades={trades} wins={wins} winrate={winrate:.1f}% total_net={total_net:.6f} avg_net={(total_net/trades if trades else 0):.6f}")
                last_summary = now

        except Exception as e:
            print("ERROR:", str(e))

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
