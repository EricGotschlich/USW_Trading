import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import pytz


ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Bitte Environment-Variablen ALPACA_API_KEY und ALPACA_SECRET_KEY setzen!")

BASE_URL = "https://data.alpaca.markets/v1beta1/news"


tz = pytz.UTC
START_DATE = datetime(2020, 1, 1, tzinfo=tz)
END_DATE   = datetime(2025, 11, 21, tzinfo=tz)

START_STR = START_DATE.strftime("%Y-%m-%dT%H:%M:%SZ")
END_STR = END_DATE.strftime("%Y-%m-%dT%H:%M:%SZ")


# Ordner dieses Skripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Projektroot = eine Ebene höher: PythonProject/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# CSV mit NASDAQ-100 Symbolen: data/nasdaq_100.csv
SYMBOLS_CSV = os.path.join(PROJECT_ROOT, "data", "nasdaq_100.csv")

# Output-Ordner für News-Parquets: data/raw/News_alpaca/
NEWS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "News_alpaca")
os.makedirs(NEWS_DIR, exist_ok=True)

# ============================
#  NASDAQ-100 TICKER LADEN
# ============================
symbols_df = pd.read_csv(SYMBOLS_CSV, encoding="latin1", engine="python")
tickers = symbols_df["Symbol"].dropna().astype(str).unique().tolist()

print(f"Gefundene NASDAQ-100 Symbole: {len(tickers)}")
print(tickers)


def fetch_news_for_symbol(symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Holt alle News für EIN Symbol im Zeitraum [start_str, end_str]
    über die Alpaca News API und gibt ein DataFrame zurück.
    """
    print(f"\n=== Hole News für {symbol} ===")
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    all_items = []
    page_token = None
    total = 0

    while True:
        params = {
            "symbols": symbol,
            "start": start_str,
            "end": end_str,
            "limit": 50,          # max. pro Request
            "sort": "desc",       # neueste zuerst
            "include_content": "false",  # HTML-Content nicht nötig
        }

        if page_token:
            params["page_token"] = page_token

        try:
            resp = requests.get(BASE_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Fehler beim Request für {symbol}: {e}")
            # bei Fehlermeldung evtl. Response-Text anzeigen
            try:
                print("Response-Text:", resp.text[:500])
            except Exception:
                pass
            break

        news_items = data.get("news", [])
        if not news_items:
            print("Keine weiteren News-Einträge.")
            break

        all_items.extend(news_items)
        total += len(news_items)
        print(f"  + {len(news_items)} News geholt (gesamt: {total})")

        page_token = data.get("next_page_token")
        if not page_token:
            break

        # kleine Pause, um API nicht zu stressen
        time.sleep(0.2)

    if not all_items:
        print(f"Keine News für {symbol} im Zeitraum gefunden.")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # Zeitstempel in richtige Datumsform bringen, falls vorhanden
    for col in ["created_at", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Symbol explizit als Spalte hinzufügen
    df["symbol"] = symbol

    return df


def main():
    print(f"\nStarte News-Download von {START_STR} bis {END_STR}")
    total_symbols = len(tickers)
    total_news = 0

    for i, symbol in enumerate(tickers, start=1):
        print(f"\n===== ({i}/{total_symbols}) {symbol} =====")
        df_symbol = fetch_news_for_symbol(symbol, START_STR, END_STR)

        if df_symbol.empty:
            print(f"--> Keine Datei für {symbol} geschrieben (keine News).")
            continue

        out_path = os.path.join(NEWS_DIR, f"{symbol}_news.parquet")
        df_symbol.to_parquet(out_path, index=False)
        total_news += len(df_symbol)

        print(f"--> {len(df_symbol)} News für {symbol} nach {out_path} geschrieben.")

    print(f"\nFERTIG. Insgesamt gespeicherte News: {total_news}")
    print(f"Parquet-Dateien liegen unter: {NEWS_DIR}")


if __name__ == "__main__":
    main()
