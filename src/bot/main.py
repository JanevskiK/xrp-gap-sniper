import os
import time
import requests

HORIZON_URL = os.getenv("HORIZON_URL", "https://horizon.stellar.org").strip()
BASE_CODE = os.getenv("BASE_CODE", "XLM").strip()
BASE_ISSUER = (os.getenv("BASE_ISSUER") or "").strip()   # empty means native
QUOTE_CODE = os.getenv("QUOTE_CODE", "USDC").strip()
QUOTE_ISSUER = (os.getenv("QUOTE_ISSUER") or "").strip()

POLL_SEC = float(os.getenv("POLL_SEC", "1.0"))

def asset_type(code: str, issuer: str) -> str:
    # Treat XLM w/ empty issuer as native
    if code.upper() == "XLM" and issuer == "":
        return "native"
    # If issuer missing for non-XLM, it's invalid
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

    # DEBUG print (you WILL see this in Railway Deploy Logs)
    print("DEBUG order_book url:", url)
    print("DEBUG order_book params:", params)

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def main():
    print(f"Horizon: {HORIZON_URL}")
    print(f"Pair: {BASE_CODE}/{QUOTE_CODE}")
    while True:
        try:
            ob = fetch_order_book()
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            print(f"bids={len(bids)} asks={len(asks)}")

        except Exception as e:
            print("ERROR:", str(e))

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
