import os

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

PARQUET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "nasdaq_news_with_sentiment.parquet"
)

print("Lese Datei:", PARQUET_PATH)
df = pd.read_parquet(PARQUET_PATH)

symbol = "ADBE"

df_sym = df[df["symbol"] == symbol]

print(f"Anzahl News für {symbol}: {len(df_sym)}\n")

print("Beispiele (ohne content):")
print(df_sym[["symbol", "id", "sentiment_score"]].head(30).to_string(index=False))

print("\nSentiment-Statistik für", symbol)
print(df_sym["sentiment_score"].describe())
