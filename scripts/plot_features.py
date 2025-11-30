"""
feature_plot.py

Kurze Analyse von Features & Targets nach dem Pre-Split-Feature-Build:

- Lädt alle *_features_with_targets.parquet aus data/processed
  (AMZN, AAPL, META, NVDA, TSLA)
- Verwendet eine kleine, feste feature_cols-Liste
- Verwendet target_cols (VWAP-basierte Return-Targets)
- Berechnet descriptive statistics für Features und Targets
- Plottet eine Feature-Korrelationsmatrix
- Meldet stark korrelierte Feature-Paare (Redundanzcheck)
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # ------------------------------------------------------------------
    # Pfade
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    images_dir = project_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dateien laden (nur Fokus-Symbole)
    # ------------------------------------------------------------------
    focus_symbols = {"AMZN", "AAPL", "META", "NVDA", "TSLA"}

    dfs = []
    for path in sorted(processed_dir.glob("*_features_with_targets.parquet")):
        symbol = path.stem.replace("_features_with_targets", "").upper()
        if symbol not in focus_symbols:
            continue

        df_sym = pd.read_parquet(path)

        # Sicherstellen, dass Index DatetimeIndex ist
        if not isinstance(df_sym.index, pd.DatetimeIndex):
            if "timestamp" in df_sym.columns:
                df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"], utc=True)
                df_sym = df_sym.set_index("timestamp")
            else:
                df_sym.index = pd.to_datetime(df_sym.index, utc=True)

        df_sym["symbol"] = symbol
        dfs.append(df_sym)

    if not dfs:
        print(f"[WARN] Keine *_features_with_targets.parquet für Fokus-Symbole in {processed_dir} gefunden.")
        return

    df = pd.concat(dfs, axis=0).sort_index()
    print(f"[INFO] Gesamt-Daten: {df.shape[0]:,} Zeilen, {df.shape[1]} Spalten")

    # ------------------------------------------------------------------
    # Features und Targets definieren
    # ------------------------------------------------------------------
    target_cols = [
        "target_return_15m",
        "target_return_30m",
        "target_return_60m",
        "target_return_120m",
    ]

    feature_cols = [
        # Trend & Returns
        "log_ret_1m",
        "log_ret_15m",
        "log_ret_60m",

        # EMA-basierte Trend-Signale
        "ema_diff_15_60",
        "ema_diff_30_120",

        # Volatilität (Aktie)
        "rv_15m",
        "rv_60m",
        "hl_span",

        # Volumen / Liquidität
        "volume_zscore_60m",
        "avg_volume_per_trade",


        # Index & relative Performance
        "index_log_ret_1m",
        "index_rv_60m",
        "rel_log_ret_60m",

        # News-Sentiment
        "effective_sentiment_t",
    ]

    # Auf vorhandene Spalten filtern (falls irgendwas fehlt)
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_cols = [c for c in target_cols if c in df.columns]

    # ------------------------------------------------------------------
    # Descriptive Statistics
    # ------------------------------------------------------------------
    feature_stats = df[feature_cols].describe().transpose()
    target_stats = df[target_cols].describe().transpose()

    feature_stats.to_csv(data_dir / "feature_descriptive_statistics.csv")
    target_stats.to_csv(data_dir / "target_descriptive_statistics.csv")

    print("[OK] Descriptive Statistics gespeichert:")
    print(f"  - {data_dir / 'feature_descriptive_statistics.csv'}")
    print(f"  - {data_dir / 'target_descriptive_statistics.csv'}")

    # ------------------------------------------------------------------
    # Feature-Korrelationsmatrix
    # ------------------------------------------------------------------
    clean_df = df[feature_cols].dropna()
    corr_df = clean_df.corr()

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        annot=False,
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title("Feature Correlation Matrix (USW_Trading)")
    plt.tight_layout()

    out_corr = images_dir / "feature_correlation_matrix.png"
    plt.savefig(out_corr)
    plt.close()
    print(f"[OK] Korrelationsmatrix gespeichert unter: {out_corr}")

    # ------------------------------------------------------------------
    # Redundanz-Check: Feature-Paare mit |corr| > 0.90
    # ------------------------------------------------------------------
    print("\n--- REDUNDANCY CHECK (|corr| > 0.90) ---")
    threshold = 0.70
    redundant_pairs = []

    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = corr_df.iloc[i, j]
            if abs(val) > threshold:
                redundant_pairs.append((c1, c2, val))

    if not redundant_pairs:
        print("Keine stark redundanten Feature-Paare gefunden.")
    else:
        for c1, c2, v in sorted(redundant_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {c1} <--> {c2} : {v:.3f}")

    print("\nAnalyse abgeschlossen.")


if __name__ == "__main__":
    main()
