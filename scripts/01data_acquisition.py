"""
This script downloads historical 1-minute adjusted OHLCV data for NASDAQ-100 symbols
using the Alpaca Market Data API and collects recent news for each symbol via Yahoo Finance (yfinance).


Outputs:
- One Parquet per symbol with 1-minute OHLCV data under <DATA_PATH>/Prices_1m_adj/
- One Parquet per symbol with raw news data under <DATA_PATH>/News_raw/

Requirements:
- Packages: alpaca-py, yfinance, pandas, pyarrow, pytz
"""

import os
from datetime import datetime
import pandas as pd
import yfinance as yf
import pytz
import yaml

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest



ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if ALPACA_API_KEY is None or ALPACA_SECRET_KEY is None:
    raise ValueError("Bitte Environment-Variablen ALPACA_API_KEY und ALPACA_SECRET_KEY setzen!")

# ----------------------------------------------------------------------
# Load data acquisition parameters from YAML configuration file
# ----------------------------------------------------------------------
# Pfad zur YAML-Datei (relativ zum Projekt-Root, wo du das Script startest)
PARAMS_PATH = os.path.join("../conf", "params.yaml")

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

data_cfg = params["DATA_ACQUISITION"]

DATA_PATH = data_cfg["DATA_PATH"]
SYMBOLS_CSV = data_cfg["SYMBOLS_CSV"]
START_DATE_STR = data_cfg["START_DATE"]
END_DATE_STR = data_cfg["END_DATE"]

START_DATE = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
END_DATE = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

# Directories for prices and news
PRICES_DIR = os.path.join(DATA_PATH, "Prices_1m_adj")
NEWS_DIR = os.path.join(DATA_PATH, "News_raw")
os.makedirs(PRICES_DIR, exist_ok=True)
os.makedirs(NEWS_DIR, exist_ok=True)

# Path to NASDAQ-100 symbols CSV (must contain column "Symbol")

# ----------------------------------------------------------------------
# Load the list of NASDAQ-100 ticker symbols
# ----------------------------------------------------------------------
ticker_list_df = pd.read_csv(SYMBOLS_CSV, encoding="latin1", engine="python")
tickers = ticker_list_df["Symbol"].dropna().astype(str).tolist()

print(f"Loaded {len(tickers)} symbols from CSV.")

# ----------------------------------------------------------------------
# Initialize Alpaca clients
# ----------------------------------------------------------------------
data_client = StockHistoricalDataClient(api_key=ALPACA_API_KEY,
                                        secret_key=ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ----------------------------------------------------------------------
# Build US trading calendar for filtering regular market hours (RTH)
# ----------------------------------------------------------------------
print(f"Fetching US trading calendar from {START_DATE_STR} to {END_DATE_STR} ...")

cal_request = GetCalendarRequest(
    start=START_DATE.date(),
    end=END_DATE.date()
)
calendar = trading_client.get_calendar(cal_request)

eastern = pytz.timezone("US/Eastern")
cal_map = {}  # date -> (open_dt, close_dt)

for c in calendar:
    # c.open / c.close are naive datetimes in US/Eastern
    open_dt = eastern.localize(c.open)
    close_dt = eastern.localize(c.close)
    cal_map[c.date] = (open_dt, close_dt)

print(f"Trading days in calendar: {len(cal_map)}")


def is_regular_trading_minute(ts: pd.Timestamp) -> bool:
    """
    Check if a given timestamp (UTC) lies within regular US market hours
    on that trading day.
    """
    # Alpaca timestamps are usually timezone-aware UTC
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")

    ts_eastern = ts_utc.tz_convert(eastern)
    d = ts_eastern.date()

    if d not in cal_map:
        return False

    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt


# ----------------------------------------------------------------------
# Helper: download 1-minute prices for one symbol via Alpaca
# ----------------------------------------------------------------------
def download_1m_prices(symbol: str) -> pd.DataFrame:
    """
    Download 1-minute OHLCV data for a single symbol using Alpaca.
    Returns a DataFrame with columns:
    timestamp (UTC), open, high, low, close, volume, vwap, trade_count, symbol.
    Only regular trading hours (RTH) are kept.
    """
    print(f"Downloading 1-minute bars for {symbol} from {START_DATE_STR} to {END_DATE_STR}")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        adjustment=Adjustment.ALL,  # Adjust for splits/dividends like "adjusted close"
        start=START_DATE,
        end=END_DATE
    )

    try:
        bars = data_client.get_stock_bars(request)
    except Exception as e:
        print(f"Error fetching bars for {symbol}: {e}")
        return pd.DataFrame()

    df = bars.df

    if df.empty:
        print(f"No bar data for {symbol}")
        return df

    # Alpaca often returns a MultiIndex (symbol, timestamp) if multiple symbols requested.
    # For our single-symbol request, we handle both cases.
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df = df.reset_index()  # 'timestamp' becomes a column

    # Ensure timestamp is timezone-aware UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Keep RTH only
    df["is_open"] = df["timestamp"].map(is_regular_trading_minute)
    df = df[df["is_open"]].drop(columns=["is_open"])

    # Add symbol column
    df["symbol"] = symbol

    # Optional: ensure column order
    cols_order = [
        "timestamp",
        "open", "high", "low", "close",
        "volume",
    ]
    # Add remaining cols (e.g. vwap, trade_count) if present
    for extra_col in ["vwap", "trade_count"]:
        if extra_col in df.columns:
            cols_order.append(extra_col)
    cols_order.append("symbol")

    df = df[cols_order]

    return df


# ----------------------------------------------------------------------
# Helper: download news for one symbol via yfinance
# ----------------------------------------------------------------------
def download_news(symbol: str) -> pd.DataFrame:
    """
    Download raw news items for a single symbol using yfinance's Ticker.news.
    Returns a DataFrame with:
    all original columns from yfinance + 'symbol'.
    """
    print(f"Fetching news for {symbol}")
    ticker = yf.Ticker(symbol)

    try:
        news_items = ticker.news
    except Exception as e:
        print(f"Warning: error while fetching news for {symbol}: {e}")
        return pd.DataFrame()

    if not news_items:
        print(f"No news items for {symbol}")
        return pd.DataFrame()

    # 1) ALLE Spalten behalten
    news_df = pd.DataFrame(news_items)

    # 2) Falls es providerPublishTime gibt → in Datum/Zeit umwandeln
    if "providerPublishTime" in news_df.columns:
        news_df["providerPublishTime"] = pd.to_datetime(
            news_df["providerPublishTime"], unit="s", utc=True
        )

    # 3) Symbol-Spalte hinzufügen
    news_df["symbol"] = symbol

    return news_df

# ----------------------------------------------------------------------
# Main loop: iterate over all NASDAQ-100 symbols
# ----------------------------------------------------------------------


counter = 0

for symbol in tickers:
    counter += 1
    print(f"\n{counter}. Processing symbol {symbol}")

    # ------------------- Prices (Alpaca, 1m) -------------------
    prices_df = download_1m_prices(symbol)
    if not prices_df.empty:
        out_prices = os.path.join(PRICES_DIR, f"{symbol}.parquet")
        prices_df.to_parquet(out_prices, index=False)
        print(f"Saved 1-minute prices to {out_prices}")
    else:
        print(f"Skipped saving prices for {symbol} (empty DataFrame)")

    # ------------------- News (Yahoo Finance) -------------------
    news_df = download_news(symbol)
    if not news_df.empty:
        out_news = os.path.join(NEWS_DIR, f"{symbol}_news.parquet")
        news_df.to_parquet(out_news, index=False)
        print(f"Saved news data to {out_news}")
    else:
        print(f"Skipped saving news for {symbol} (no news).")
print("\nData acquisition finished for all NASDAQ-100 symbols.")
