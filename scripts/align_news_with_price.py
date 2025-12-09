"""
Aligns news sentiment with 1-minute price data for ALL symbols
and applies an exponential time-decay to the sentiment.

For each 1-minute bar we compute:
- last_news_sentiment: sentiment of the most recent news before this bar
- news_age_minutes: age of that news at this bar (in minutes)
- effective_sentiment_t: time-decayed sentiment score
      effective_sentiment_t = sentiment * exp(-LAMBDA * news_age_minutes)
- news_id / news_headline (optional, falls im News-File vorhanden)

Input (relative to project root):
- data/processed/nasdaq_news_with_sentiment.parquet
  expected cols :
    ['symbol', 'sentiment_score', <timestamp col>, 'id', 'headline'/ 'title', ...]

- data/raw/Prices_1m_adj/<SYMBOL>.parquet
  expected cols:
    ['timestamp', 'open', 'high', 'low', 'close', 'volume', ...]

Output:
- data/processed/<SYMBOL>_aligned_with_news.parquet   (für jedes Symbol)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------- CONFIG ----------------- #

# Zerfallsparameter λ:
#  hoher Wert  -> News-Effekt verschwindet schnell
#  kleiner Wert -> News wirkt lange nach
# Half-Life (in Minuten) ≈ ln(2) / λ
LAMBDA = 0.001  # ~693 Minuten (~11,5 Stunden) Halbwertszeit


# ----------------- Hilfsfunktion pro Symbol ----------------- #

def align_symbol(
    symbol: str,
    prices_path: Path,
    news_all: pd.DataFrame,
    ts_col: str,
    headline_col: str | None,
    out_dir: Path,
) -> None:
    """Aligniere News-Sentiment mit 1-Minute-Preisen für EIN Symbol."""

    if not prices_path.exists():
        print(f"[WARN] Price file missing for {symbol}: {prices_path}")
        return

    # --- News für dieses Symbol --- #
    news_sym = news_all[news_all["symbol"] == symbol].copy()
    if news_sym.empty:
        print(f"[WARN] No news for symbol {symbol}, skipping.")
        return

    news_sym[ts_col] = pd.to_datetime(news_sym[ts_col], utc=True, errors="coerce")
    news_sym = news_sym.dropna(subset=[ts_col])
    news_sym = news_sym.sort_values(ts_col).reset_index(drop=True)

    # --- Preise für dieses Symbol --- #
    prices = pd.read_parquet(prices_path)

    if "timestamp" not in prices.columns:
        raise ValueError(f"{prices_path} has no 'timestamp' column.")

    prices["timestamp"] = pd.to_datetime(
        prices["timestamp"], utc=True, errors="coerce"
    )
    prices = prices.dropna(subset=["timestamp"])
    prices = prices.sort_values("timestamp").reset_index(drop=True)

    print(
        f"  [INFO] {symbol}: {len(prices):,} bars  "
        f"({prices['timestamp'].min()} .. {prices['timestamp'].max()}), "
        f"{len(news_sym):,} news items"
    )

    # --- Spalten für Alignment vorbereiten --- #
    prices["last_news_sentiment"] = 0.0
    prices["news_age_minutes"] = np.nan
    prices["effective_sentiment_t"] = 0.0
    prices["news_id"] = None
    if headline_col is not None:
        prices["news_headline"] = None

    # --- Hauptloop: News -> Bars --- #
    news_idx = 0

    for i, row in prices.iterrows():
        current_time = row["timestamp"]

        # News-Zeiger nach vorne bewegen, solange nächste News <= current_time
        while (
            news_idx < len(news_sym) - 1
            and news_sym.iloc[news_idx + 1][ts_col] <= current_time
        ):
            news_idx += 1

        # Falls mindestens eine News vor diesem Zeitpunkt liegt:
        if news_sym.iloc[news_idx][ts_col] <= current_time:
            nrow = news_sym.iloc[news_idx]

            news_age = (current_time - nrow[ts_col]).total_seconds() / 60.0
            sentiment = float(nrow["sentiment_score"])
            effective = sentiment * np.exp(-LAMBDA * news_age)

            prices.at[i, "last_news_sentiment"] = sentiment
            prices.at[i, "news_age_minutes"] = news_age
            prices.at[i, "effective_sentiment_t"] = effective

            if "id" in news_sym.columns:
                prices.at[i, "news_id"] = nrow["id"]
            if headline_col is not None:
                prices.at[i, "news_headline"] = nrow[headline_col]

        if (i + 1) % 100_000 == 0:
            print(
                f"    processed {i + 1:,} / {len(prices):,} bars "
                f"({(i + 1) / len(prices) * 100:.1f}%)"
            )

    # --- Speichern & kleine Statistik --- #
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_aligned_with_news.parquet"
    prices.to_parquet(out_path, index=False)

    has_news = prices["news_age_minutes"].notna()
    total_bars = len(prices)

    print(f"  [OK] Saved aligned data for {symbol} -> {out_path}")
    print(
        f"      bars with news: {has_news.sum():,} "
        f"({has_news.sum() / total_bars * 100:.1f}%), "
        f"without news: {(~has_news).sum():,}"
    )


# ----------------- main ----------------- #

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    prices_dir = data_dir / "raw" / "Prices_1m_adj"
    out_dir = data_dir / "processed"

    news_path = data_dir / "processed" / "nasdaq_news_with_sentiment.parquet"

    if not news_path.exists():
        raise FileNotFoundError(f"News parquet not found: {news_path}")
    if not prices_dir.exists():
        raise FileNotFoundError(f"Prices directory not found: {prices_dir}")

    print(f"[INFO] Loading news parquet: {news_path}")
    news_all = pd.read_parquet(news_path)

    # Pflichtspalten prüfen
    if "symbol" not in news_all.columns:
        raise ValueError("News parquet must contain 'symbol' column.")
    if "sentiment_score" not in news_all.columns:
        raise ValueError("News parquet must contain 'sentiment_score' column.")

    # Zeitspalte finden (created_at / timestamp / published_at)
    ts_col_candidates = ["created_at", "timestamp", "published_at"]
    ts_col = None
    for c in ts_col_candidates:
        if c in news_all.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found in news parquet. "
            f"Tried: {ts_col_candidates}"
        )

    # Headline-Spalte (falls vorhanden – nur für Info / Debug)
    headline_col = None
    for c in ["headline", "title"]:
        if c in news_all.columns:
            headline_col = c
            break

    print(f"[INFO] Using news timestamp column: {ts_col}")
    if headline_col:
        print(f"[INFO] Using headline column: {headline_col}")

    # Liste aller verfügbaren Symbole aus Price-Parquets
    price_files = sorted(prices_dir.glob("*.parquet"))
    symbols = [f.stem for f in price_files]

    print(f"\n[INFO] Found {len(symbols)} symbols in {prices_dir}:")
    print(", ".join(symbols))

    half_life = np.log(2) / LAMBDA
    print(
        f"\n[INFO] Decay parameter λ = {LAMBDA} "
        f"(half-life ≈ {half_life:.1f} min ≈ {half_life / 60:.1f} h)\n"
    )

    # Für jedes Symbol alignen
    for symbol, prices_path in zip(symbols, price_files):
        print("\n" + "-" * 60)
        print(f"[SYMBOL] {symbol}")
        print("-" * 60)
        try:
            align_symbol(
                symbol=symbol,
                prices_path=prices_path,
                news_all=news_all,
                ts_col=ts_col,
                headline_col=headline_col,
                out_dir=out_dir,
            )
        except Exception as e:
            print(f"  [ERROR] Failed for {symbol}: {e}")

    print("\n" + "=" * 60)
    print("FINISHED ALIGNING NEWS FOR ALL SYMBOLS")
    print("=" * 60)


if __name__ == "__main__":
    main()
