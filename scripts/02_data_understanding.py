"""
02_data_understanding.py

Data Understanding

Nutzt:
- data/processed/nasdaq_news_with_sentiment.parquet
- data/raw/Prices_1m_adj/<SYMBOL>.parquet  (1-Minuten OHLCV + trade_count)

Erzeugt Plots (im Ordner images/):
- sentiment_distribution.png
- <STOCK_SYMBOL>_price.png
- <STOCK_SYMBOL>_volume.png
- <STOCK_SYMBOL>_trades.png
- QQQ_price.png
- QQQ_volume.png
- QQQ_trades.png
- <STOCK_SYMBOL>_abnormal_vs_QQQ.png   <= NEU
"""

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
STOCK_SYMBOL = "AAPL"          # normale Aktie
INDEX_SYMBOL = "QQQ"           # ETF / Index
NEWS_FILE_NAME = "nasdaq_news_with_sentiment.parquet"

# Zeitfenster (in Minuten) für Event-Study
WIN_BEFORE = 30
WIN_AFTER = 30
MAX_EVENTS_PER_CAT = 1000

# Sentiment-Schwellen
VERY_NEG = -0.75
NEG = -0.25
POS = 0.25
VERY_POS = 0.75


# ---------------------------------------------------------
# Pfade
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "Prices_1m_adj"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Hilfsfunktionen: Laden & Aufbereiten
# ---------------------------------------------------------
def categorize_sentiment(score: float) -> str:
    """Einstufung in 5 Sentiment-Buckets."""
    if score <= VERY_NEG:
        return "Very negative"
    if score <= NEG:
        return "Negative"
    if score < POS:
        return "Neutral"
    if score < VERY_POS:
        return "Positive"
    return "Very positive"


def load_news_data() -> pd.DataFrame:
    """Lädt News + Sentiment, bereitet Timestamps & Kategorien vor."""
    path = PROCESSED_DIR / NEWS_FILE_NAME
    print(f"[INFO] Lade News mit Sentiment aus {path}")
    if not path.exists():
        raise FileNotFoundError(f"Sentiment-Datei nicht gefunden: {path}")

    df = pd.read_parquet(path)

    if "sentiment_score" not in df.columns:
        raise ValueError("Spalte 'sentiment_score' fehlt in der News-Datei.")

    # Zeitstempel finden (Alpaca: 'created_at')
    ts_col = None
    for cand in ["created_at", "providerPublishTime", "timestamp", "time"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError("Keine Zeitstempel-Spalte in News gefunden.")

    if ts_col == "providerPublishTime":
        df[ts_col] = pd.to_datetime(df[ts_col], unit="s", utc=True, errors="coerce")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    df = df.rename(columns={ts_col: "created_at"})
    df = df.dropna(subset=["created_at"])

    # Sentiment-Buckets
    df["sentiment_bucket"] = df["sentiment_score"].apply(categorize_sentiment)

    print("\n[INFO] Sentiment-Bucket-Verteilung (alle Symbole):")
    print(df["sentiment_bucket"].value_counts().sort_index())

    # Optional: kleine Statistik
    print("\n[INFO] Sentiment-Score-Statistik:")
    print(df["sentiment_score"].describe())

    return df


def load_price_data(symbol: str) -> pd.DataFrame:
    """
    Lädt 1-Minuten-Preisdaten für ein Symbol und filtert auf RTH (09:30–16:00 ET).
    Erwartet Spalten: timestamp, close, volume, trade_count (trade_count optional).
    """
    path = RAW_PRICES_DIR / f"{symbol}.parquet"
    print(f"[INFO] Lade Preisdaten für {symbol} aus {path}")
    if not path.exists():
        raise FileNotFoundError(f"Preisdatei nicht gefunden: {path}")

    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{symbol}: Spalte 'timestamp' fehlt.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    # US/Eastern für Handelszeiten
    df["timestamp_et"] = df.index.tz_convert("US/Eastern")
    mask_rth = df["timestamp_et"].dt.time.between(time(9, 30), time(16, 0))
    df = df.loc[mask_rth].copy()
    df.drop(columns=["timestamp_et"], inplace=True)

    print(f"  [INFO] {symbol}: {len(df):,} 1-Minuten-Bars nach RTH-Filter")
    return df


# ---------------------------------------------------------
# Sentiment-Verteilung
# ---------------------------------------------------------
def plot_sentiment_distribution(news_df: pd.DataFrame) -> None:
    order = [
        "Very negative",
        "Negative",
        "Neutral",
        "Positive",
        "Very positive",
    ]
    counts = news_df["sentiment_bucket"].value_counts().reindex(order).fillna(0)
    total = counts.sum()
    percentages = counts / total * 100

    plt.figure(figsize=(8, 5))
    bars = plt.bar(order, counts.values, alpha=0.8)

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title("Sentiment Distribution (all NASDAQ-100 news)")
    plt.ylabel("Number of articles")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out = IMAGES_DIR / "sentiment_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved: {out}")


# ---------------------------------------------------------
# Event-Study-Helfer
# ---------------------------------------------------------
def _filter_events_to_intraday_window(
    news_subset: pd.DataFrame,
    window_before: int,
    window_after: int,
) -> pd.DataFrame:
    """
    Behalte nur News, bei denen [-window_before, +window_after] komplett in
    den Regular Trading Hours (09:30–16:00 ET) liegt.
    """
    if news_subset.empty:
        return news_subset

    news_subset = news_subset.copy()
    created_et = news_subset["created_at"].dt.tz_convert("US/Eastern")
    news_subset["created_at_et"] = created_et

    start_allowed = time(9, 30 + window_before // 60)  # grobe Untergrenze
    end_allowed = time(16 - window_after // 60, 0)     # grobe Obergrenze

    mask = news_subset["created_at_et"].dt.time.between(start_allowed, end_allowed)
    filtered = news_subset.loc[mask].copy()

    print(
        f"    [INFO] Intraday-Filter: {len(filtered)}/{len(news_subset)} Events "
        f"behalten (Fenster {window_before}m vor, {window_after}m nach News)"
    )
    return filtered


def filter_isolated_events_for_symbol(
    news_symbol: pd.DataFrame,
    window_minutes: int,
) -> pd.DataFrame:
    """
    Behalte nur News-Events eines Symbols, bei denen KEINE andere News
    desselben Symbols im Zeitfenster ± window_minutes Minuten liegt.

    Ziel: Event-Fenster sollen nicht von weiteren News überlagert werden.
    """
    if news_symbol.empty:
        return news_symbol

    news_symbol = news_symbol.sort_values("created_at").copy()
    times = news_symbol["created_at"].to_numpy()
    n = len(news_symbol)

    keep = np.ones(n, dtype=bool)
    delta = np.timedelta64(window_minutes, "m")

    for i in range(n):
        # Abstand zur vorherigen News
        if i > 0 and times[i] - times[i - 1] <= delta:
            keep[i] = False
        # Abstand zur nächsten News
        if i < n - 1 and times[i + 1] - times[i] <= delta:
            keep[i] = False

    filtered = news_symbol.iloc[keep].copy()
    print(
        f"    [INFO] Isolated-event filter (±{window_minutes} min): "
        f"kept {len(filtered)}/{n} events for symbol "
        f"{filtered['symbol'].iloc[0] if len(filtered) else 'N/A'}"
    )
    return filtered


def compute_price_path(
    news_subset: pd.DataFrame,
    price_df: pd.DataFrame,
    window_before: int = WIN_BEFORE,
    window_after: int = WIN_AFTER,
    max_events: int = MAX_EVENTS_PER_CAT,
) -> np.ndarray:
    """
    Durchschnittlicher Preis-Pfad in % relativ zu Preis bei t=0.
    """
    target_len = window_before + window_after + 1
    if news_subset.empty:
        return np.zeros(target_len)

    news_subset = _filter_events_to_intraday_window(
        news_subset, window_before, window_after
    )
    if news_subset.empty:
        return np.zeros(target_len)

    n_sample = min(max_events, len(news_subset))
    events = news_subset.sample(n=n_sample, random_state=42)

    paths = []

    for _, ev in events.iterrows():
        event_time = ev["created_at"].floor("min")
        start_time = event_time - pd.Timedelta(minutes=window_before)
        end_time = event_time + pd.Timedelta(minutes=window_after)

        window = price_df.loc[start_time:end_time, ["close"]].copy()
        if window.empty:
            continue

        # 1-Minuten-Raster
        window = (
            window
            .resample("1min")
            .last()
        )

        full_index = pd.date_range(
            start=start_time, end=end_time, freq="1min", tz="UTC"
        )
        window = window.reindex(full_index).ffill()

        if window["close"].isna().any():
            continue

        values = window["close"].values
        if len(values) != target_len:
            continue

        base = values[window_before]
        if base <= 0 or np.isnan(base):
            continue

        rel = (values / base - 1.0) * 100.0
        paths.append(rel)

    if not paths:
        print("    [WARN] Keine gültigen Preisfenster, gebe Nullen zurück.")
        return np.zeros(target_len)

    paths_arr = np.vstack(paths)
    print(f"    [INFO] Price: nutze {paths_arr.shape[0]} gültige Events")
    return paths_arr.mean(axis=0)


def compute_liquidity_path(
    news_subset: pd.DataFrame,
    price_df: pd.DataFrame,
    value_col: str,
    window_before: int = WIN_BEFORE,
    window_after: int = WIN_AFTER,
    baseline_gap: int = 10,          # letzte 10 Minuten vor Event nicht in Baseline
    max_events: int = MAX_EVENTS_PER_CAT,
    min_baseline: float | None = None,
    clip_pct: float | None = 300.0,  # relative Änderung auf ±clip_pct % begrenzen
) -> np.ndarray:
    """
    Volume/Trade:
    relative Abweichung vom Durchschnittslevel vor der News.

    Baseline = mittlerer Wert von [-window_before, -(baseline_gap)] Minuten.
    Events mit extrem niedriger Baseline werden verworfen, damit
    keine absurden Prozentwerte entstehen.
    """
    target_len = window_before + window_after + 1

    if news_subset.empty or value_col not in price_df.columns:
        return np.zeros(target_len)

    # Nur Events behalten, bei denen das ganze Zeitfenster im Handel liegt
    news_subset = _filter_events_to_intraday_window(
        news_subset, window_before, window_after
    )
    if news_subset.empty:
        return np.zeros(target_len)

    # Mindest-Baseline: Standard = 10 % des globalen Medians
    if min_baseline is None:
        global_median = price_df[value_col].median()
        # Falls alles sehr klein ist, nicht auf 0 fallen lassen
        min_baseline = max(global_median * 0.1, 1.0)

    n_sample = min(max_events, len(news_subset))
    events = news_subset.sample(n=n_sample, random_state=42)

    paths = []

    for _, ev in events.iterrows():
        event_time = ev["created_at"].floor("min")
        start_time = event_time - pd.Timedelta(minutes=window_before)
        end_time = event_time + pd.Timedelta(minutes=window_after)

        window = price_df.loc[start_time:end_time, [value_col]].copy()
        if window.empty:
            continue

        # 1-Minuten-Aggregation
        window = (
            window
            .resample("1min")
            .sum()
        )

        full_index = pd.date_range(
            start=start_time, end=end_time, freq="1min", tz="UTC"
        )
        window = window.reindex(full_index).fillna(0.0)

        values = window[value_col].values
        if len(values) != target_len:
            continue

        # Baseline: [-window_before, -baseline_gap]
        baseline_slice = values[: window_before - baseline_gap]
        baseline_mean = baseline_slice.mean()

        # Events mit zu niedriger Baseline verwerfen
        if baseline_mean <= min_baseline or np.isnan(baseline_mean):
            continue

        rel = (values / baseline_mean - 1.0) * 100.0

        # Extremwerte begrenzen, damit der Plot lesbar bleibt
        if clip_pct is not None:
            rel = np.clip(rel, -clip_pct, clip_pct)

        paths.append(rel)

    if not paths:
        print(f"    [WARN] Keine gültigen {value_col}-Fenster, gebe Nullen zurück.")
        return np.zeros(target_len)

    paths_arr = np.vstack(paths)
    print(
        f"    [INFO] {value_col}: nutze {paths_arr.shape[0]} gültige Events, "
        f"min_baseline={min_baseline:.2f}"
    )
    return paths_arr.mean(axis=0)


# ---------------------------------------------------------
# NEU: Abnormale Preisreaktion (Stock - Index)
# ---------------------------------------------------------
def compute_abnormal_price_path(
    news_subset: pd.DataFrame,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    window_before: int = WIN_BEFORE,
    window_after: int = WIN_AFTER,
    max_events: int = MAX_EVENTS_PER_CAT,
) -> np.ndarray:
    """
    Berechnet durchschnittliche *abnormale* Preisreaktion:

      abnormal(t) = (Stock-Return in %) - (Index-Return in %)

    für das Fenster [-window_before, +window_after] relativ zur Newszeit.

    Idee: Makro-/Marktbewegungen (z.B. QQQ) werden rausgerechnet, um den
    stock-spezifischen Effekt besser zu sehen.
    """
    target_len = window_before + window_after + 1
    if news_subset.empty:
        return np.zeros(target_len)

    # Nur News mit komplettem Intraday-Fenster
    news_subset = _filter_events_to_intraday_window(
        news_subset, window_before, window_after
    )
    if news_subset.empty:
        return np.zeros(target_len)

    n_sample = min(max_events, len(news_subset))
    events = news_subset.sample(n=n_sample, random_state=42)

    paths = []

    for _, ev in events.iterrows():
        event_time = ev["created_at"].floor("min")
        start_time = event_time - pd.Timedelta(minutes=window_before)
        end_time = event_time + pd.Timedelta(minutes=window_after)

        # Fenster für Stock und Index
        win_stock = stock_df.loc[start_time:end_time, ["close"]].copy()
        win_index = index_df.loc[start_time:end_time, ["close"]].copy()
        if win_stock.empty or win_index.empty:
            continue

        # 1-Minuten-Raster
        win_stock = win_stock.resample("1min").last()
        win_index = win_index.resample("1min").last()

        # Gleicher vollständiger Zeitindex
        full_index = pd.date_range(
            start=start_time, end=end_time, freq="1min", tz="UTC"
        )
        win_stock = win_stock.reindex(full_index).ffill()
        win_index = win_index.reindex(full_index).ffill()

        if win_stock["close"].isna().any() or win_index["close"].isna().any():
            continue

        stock_vals = win_stock["close"].values
        index_vals = win_index["close"].values
        if len(stock_vals) != target_len or len(index_vals) != target_len:
            continue

        # Basispreise bei t=0 (Eventzeit)
        base_stock = stock_vals[window_before]
        base_index = index_vals[window_before]
        if base_stock <= 0 or base_index <= 0 or np.isnan(base_stock) or np.isnan(base_index):
            continue

        # relative Preisänderung in %
        stock_rel = (stock_vals / base_stock - 1.0) * 100.0
        index_rel = (index_vals / base_index - 1.0) * 100.0

        abnormal = stock_rel - index_rel
        paths.append(abnormal)

    if not paths:
        print("    [WARN] Keine gültigen abnormal-price-Fenster, gebe Nullen zurück.")
        return np.zeros(target_len)

    paths_arr = np.vstack(paths)
    print(f"    [INFO] Abnormal Price: nutze {paths_arr.shape[0]} gültige Events")
    return paths_arr.mean(axis=0)


def plot_abnormal_price_for_symbol(
    symbol: str,
    index_symbol: str,
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> None:
    """
    Plot der durchschnittlichen *abnormalen* Preisreaktion
    (Stock% - Index%) nach Sentiment-Buckets.

    So sieht man, ob z.B. sehr positive News systematisch zu
    Über-/Unterperformance relativ zum NASDAQ-100 (QQQ) führen.
    """
    news_symbol = news_df[news_df["symbol"] == symbol]
    if news_symbol.empty:
        print(f"[WARN] Keine News für {symbol} gefunden – skip abnormal price plot.")
        return

    # Nur isolierte Events ohne weitere News im ±WIN_AFTER-Fenster
    news_symbol = filter_isolated_events_for_symbol(news_symbol, WIN_AFTER)
    if news_symbol.empty:
        print(f"[WARN] Nach Isolated-Filter keine News mehr für {symbol} – skip abnormal price plot.")
        return

    print(
        f"\n[INFO] Abnormal Price Event Study für {symbol} vs. {index_symbol} "
        f"(-{WIN_BEFORE} bis +{WIN_AFTER} Minuten, nach Sentiment)"
    )

    minutes = np.arange(-WIN_BEFORE, WIN_AFTER + 1)

    # 5 Sentiment-Buckets wie in den anderen Plots
    buckets: dict[str, pd.DataFrame] = {
        "Very positive (≥ 0.75)": news_symbol[news_symbol["sentiment_bucket"] == "Very positive"],
        "Positive (0.25–0.75)":   news_symbol[news_symbol["sentiment_bucket"] == "Positive"],
        "Neutral (−0.25–0.25)":   news_symbol[news_symbol["sentiment_bucket"] == "Neutral"],
        "Negative (−0.75–−0.25)": news_symbol[news_symbol["sentiment_bucket"] == "Negative"],
        "Very negative (≤ −0.75)": news_symbol[news_symbol["sentiment_bucket"] == "Very negative"],
    }

    plt.figure(figsize=(12, 6))

    for label, subset in buckets.items():
        # zu wenige Events = sehr noisy -> skip
        if len(subset) < 20:
            print(f"    [INFO] {label}: nur {len(subset)} Events – übersprungen.")
            continue

        path = compute_abnormal_price_path(subset, stock_df, index_df)

        # Wenn alles ~0 ist UND wir wenig Events haben, ist das meist „kein Signal“
        if np.allclose(path, 0):
            print(f"    [INFO] {label}: path ~0 – vermutlich keine gültigen Fenster, übersprungen.")
            continue

        plt.plot(minutes, path, linewidth=1.8, label=label)

    plt.axvline(0, linestyle=":", linewidth=1.5, color="black", label="News time")
    plt.axhline(0, linewidth=0.8, color="black")

    # Post-News-Fenster leicht einfärben
    plt.axvspan(0, WIN_AFTER, alpha=0.05)

    plt.title(
        f"{symbol}: Abnormal price reaction vs. {index_symbol} "
        f"by news sentiment (-{WIN_BEFORE} to +{WIN_AFTER} min)"
    )
    plt.xlabel("Minutes relative to news publication")
    plt.ylabel("Average abnormal price change (%)\n(Stock minus index)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = IMAGES_DIR / f"{symbol}_abnormal_vs_{index_symbol}_price.png"
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved: {out}")


# ---------------------------------------------------------
# Plot-Funktionen (Preis, Volumen, Trades)
# ---------------------------------------------------------
def plot_price_for_symbol(
    symbol: str,
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> None:
    news_symbol = news_df[news_df["symbol"] == symbol]
    if news_symbol.empty:
        print(f"[WARN] Keine News für {symbol} gefunden – skip price event study.")
        return

    # Nur isolierte Events ohne weitere News im ±WIN_AFTER-Fenster
    news_symbol = filter_isolated_events_for_symbol(news_symbol, WIN_AFTER)
    if news_symbol.empty:
        print(f"[WARN] Nach Isolated-Filter keine News mehr für {symbol} – skip price event study.")
        return

    print(f"\n[INFO] Price Event Study für {symbol} (-{WIN_BEFORE} bis +{WIN_AFTER} Minuten)")

    vn = news_symbol[news_symbol["sentiment_bucket"] == "Very negative"]
    neg = news_symbol[news_symbol["sentiment_bucket"] == "Negative"]
    neu = news_symbol[news_symbol["sentiment_bucket"] == "Neutral"]
    pos = news_symbol[news_symbol["sentiment_bucket"] == "Positive"]
    vp = news_symbol[news_symbol["sentiment_bucket"] == "Very positive"]

    minutes = np.arange(-WIN_BEFORE, WIN_AFTER + 1)

    path_vn = compute_price_path(vn, price_df)
    path_neg = compute_price_path(neg, price_df)
    path_neu = compute_price_path(neu, price_df)
    path_pos = compute_price_path(pos, price_df)
    path_vp = compute_price_path(vp, price_df)

    plt.figure(figsize=(12, 6))
    # andere Styles als im Beispielprojekt
    plt.plot(minutes, path_vp, marker="o", markevery=12, linewidth=2.2, label="Very positive (≥ 0.75)")
    plt.plot(minutes, path_pos, linestyle="-.", linewidth=1.8, label="Positive (0.25–0.75)")
    plt.plot(minutes, path_neu, linestyle=":", linewidth=1.8, label="Neutral (−0.25–0.25)")
    plt.plot(minutes, path_neg, linestyle="--", linewidth=1.8, label="Negative (−0.25–−0.75)")
    plt.plot(minutes, path_vn, marker="x", markevery=12, linewidth=2.2, label="Very negative (≤ −0.75)")

    # Newszeit markieren
    plt.axvline(0, linestyle=":", linewidth=1.5, color="black")
    # Post-News-Fenster dezent einfärben
    plt.axvspan(0, WIN_AFTER, alpha=0.05)

    plt.title(f"{symbol}: Average price reaction around own news (-{WIN_BEFORE} to +{WIN_AFTER} min)")
    plt.xlabel("Minutes relative to news publication")
    plt.ylabel("Average price change (%)")
    plt.axhline(0, linewidth=0.5, color="black")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = IMAGES_DIR / f"{symbol}_price_event_study.png"
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved: {out}")


def plot_liquidity_for_symbol(
    symbol: str,
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    value_col: str,
    label_y: str,
) -> None:
    """Event-Study für Volumen / Trade Count mit 5 Sentiment-Buckets."""
    if value_col not in price_df.columns:
        print(f"[WARN] Spalte '{value_col}' fehlt für {symbol} – überspringe.")
        return

    news_symbol = news_df[news_df["symbol"] == symbol]
    if news_symbol.empty:
        print(f"[WARN] Keine News für {symbol} – überspringe {value_col}-Event-Study.")
        return

    # Nur isolierte Events verwenden
    news_symbol = filter_isolated_events_for_symbol(news_symbol, WIN_AFTER)
    if news_symbol.empty:
        print(
            f"[WARN] Nach Isolated-Filter keine News mehr für {symbol} – "
            f"überspringe {value_col}."
        )
        return

    print(
        f"\n[INFO] {value_col.capitalize()} Event Study für {symbol} "
        f"(-{WIN_BEFORE} bis +{WIN_AFTER} Minuten)"
    )

    # 5 Buckets wie im Price-Plot
    vn = news_symbol[news_symbol["sentiment_bucket"] == "Very negative"]
    neg = news_symbol[news_symbol["sentiment_bucket"] == "Negative"]
    neu = news_symbol[news_symbol["sentiment_bucket"] == "Neutral"]
    pos = news_symbol[news_symbol["sentiment_bucket"] == "Positive"]
    vp = news_symbol[news_symbol["sentiment_bucket"] == "Very positive"]

    minutes = np.arange(-WIN_BEFORE, WIN_AFTER + 1)

    # Mindest-Baseline aus den Daten ableiten
    median_val = price_df[value_col].median()
    min_baseline = max(median_val * 0.1, 1.0)

    path_vn = compute_liquidity_path(vn, price_df, value_col, min_baseline=min_baseline)
    path_neg = compute_liquidity_path(neg, price_df, value_col, min_baseline=min_baseline)
    path_neu = compute_liquidity_path(neu, price_df, value_col, min_baseline=min_baseline)
    path_pos = compute_liquidity_path(pos, price_df, value_col, min_baseline=min_baseline)
    path_vp = compute_liquidity_path(vp, price_df, value_col, min_baseline=min_baseline)

    plt.figure(figsize=(12, 6))
    plt.plot(minutes, path_vp, marker="o", markevery=10,
             linewidth=2.0, label="Very positive (≥ 0.75)")
    plt.plot(minutes, path_pos, linestyle="-.", linewidth=1.8,
             label="Positive (0.25–0.75)")
    plt.plot(minutes, path_neu, linestyle=":", linewidth=1.8,
             label="Neutral (−0.25–0.75)")
    plt.plot(minutes, path_neg, linestyle="--", linewidth=1.8,
             label="Negative (−0.75–−0.25)")
    plt.plot(minutes, path_vn, marker="x", markevery=10,
             linewidth=2.0, label="Very negative (≤ −0.25)")

    plt.axvline(0, linestyle=":", linewidth=1.5, color="black")
    plt.axvspan(0, WIN_AFTER, alpha=0.05)

    plt.title(
        f"{symbol}: {label_y} reaction around own news "
        f"(-{WIN_BEFORE} to +{WIN_AFTER} min)"
    )
    plt.xlabel("Minutes relative to news publication")
    plt.ylabel(f"Average change vs. pre-news baseline (%)\n({label_y})")
    plt.axhline(0, linewidth=0.5, color="black")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = IMAGES_DIR / f"{symbol}_{value_col}.png"
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved: {out}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("DATA UNDERSTANDING (STOCK + QQQ)")
    print("=" * 60 + "\n")

    news_df = load_news_data()
    price_stock = load_price_data(STOCK_SYMBOL)
    price_index = load_price_data(INDEX_SYMBOL)

    # 1) Sentiment-Verteilung über alle News
    plot_sentiment_distribution(news_df)

    # 2) Event-Studys für Einzelaktie
    plot_price_for_symbol(STOCK_SYMBOL, news_df, price_stock)
    plot_liquidity_for_symbol(
        STOCK_SYMBOL, news_df, price_stock, value_col="volume", label_y="Volume"
    )
    plot_liquidity_for_symbol(
        STOCK_SYMBOL, news_df, price_stock, value_col="trade_count", label_y="Trade count"
    )

    # 2b) NEU: Abnormale Preisreaktion (Stock – QQQ)
    plot_abnormal_price_for_symbol(
        STOCK_SYMBOL, INDEX_SYMBOL, news_df, price_stock, price_index
    )



    print("\n" + "=" * 60)
    print("Fertig. Schau in den 'images' Ordner für alle Grafiken.")
    print("Zusätzlich neu:")
    print(f"  - {STOCK_SYMBOL}_abnormal_vs_{INDEX_SYMBOL}_price.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
