# Directory structure:
# alpaca_scanner_app/
# ├── app.py                 ← main entry
# ├── strategy.py            ← your updated strategy file (already uploaded)
# └── top_volume.py          ← fetches top 10 volume stocks from FMP

# ---------- top_volume.py ----------
import requests

def get_top_volume_tickers():
    url = "https://financialmodelingprep.com/api/v3/stock_market/actives"
    params = {"apikey": "anc5nLqF1PuZUQGhxDHpgXuU0Yp9Cj0V"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return [item["symbol"] for item in data[:10]]
    except Exception as e:
        print("Error fetching top volume tickers:", e)
        return []

# ---------- app.py ----------
from strategy import scan_strategies
from top_volume import get_top_volume_tickers

def main():
    print("Fetching top volume tickers...")
    tickers = get_top_volume_tickers()
    print("Scanning strategies on:", tickers)

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        result = scan_strategies(ticker)
        for strat, outcome in result.items():
            if outcome:
                print(f"{strat}: {outcome}")

if __name__ == "__main__":
    main()
