# scripts/split_dataset.py

from pathlib import Path
from typing import List

import pandas as pd
import yaml

# ------------------------------------------------------------
# 1) Welche Features / Targets ins ML-Modell sollen
# ------------------------------------------------------------

FEATURE_COLS: List[str] = [
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

    # Volumen / Liquidität
    "volume_zscore_60m",
    "avg_volume_per_trade",
    "hl_span",

    # Index & relative Performance
    "index_log_ret_1m",
    "index_rv_60m",
    "rel_log_ret_60m",

    # News-Sentiment
    "last_news_sentiment",
    "news_age_minutes",
    "effective_sentiment_t",
]

TARGET_COLS: List[str] = [
    "target_return_15m",
    "target_return_30m",
    "target_return_60m",
    "target_return_120m",
]

# Nur diese Symbole sollen berücksichtigt werden
ALLOWED_SYMBOLS = {"AMZN", "AAPL", "META", "NVDA", "TSLA"}


# ------------------------------------------------------------
# 2) DATA_PREP-Daten (Train/Val/Test) aus params.yaml lesen
# ------------------------------------------------------------

def load_data_prep_dates(project_root: Path):
    conf_path = project_root / "conf" / "params.yaml"
    if not conf_path.exists():
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_prep_cfg = params["DATA_PREP"]
    train_date = pd.to_datetime(data_prep_cfg["TRAIN_DATE"], utc=True)
    val_date = pd.to_datetime(data_prep_cfg["VALIDATION_DATE"], utc=True)
    test_date = pd.to_datetime(data_prep_cfg["TEST_DATE"], utc=True)

    return train_date, val_date, test_date


# ------------------------------------------------------------
# 3) Hauptfunktion
# ------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"

    # 3.1 Data-Prep-Grenzen laden
    train_date, val_date, test_date = load_data_prep_dates(project_root)
    print(
        f"[INFO] Data-Prep-Grenzen:\n"
        f"  TRAIN_DATE      = {train_date}\n"
        f"  VALIDATION_DATE = {val_date}\n"
        f"  TEST_DATE       = {test_date}"
    )

    # 3.2 Alle Feature+Target-Dateien holen
    all_files = sorted(processed_dir.glob("*_features_with_targets.parquet"))
    if not all_files:
        print(f"[WARN] Keine *_features_with_targets.parquet in {processed_dir} gefunden.")
        return

    # *** HIER: explizit auf deine 5 Symbole filtern ***
    feature_target_files = []
    for p in all_files:
        symbol = p.stem.replace("_features_with_targets", "").upper()
        if symbol in ALLOWED_SYMBOLS:
            feature_target_files.append(p)

    if not feature_target_files:
        print(f"[WARN] Keine passenden Dateien für {ALLOWED_SYMBOLS} gefunden.")
        return

    print(f"[INFO] Verwende folgende Feature+Target-Dateien (5 Symbole):")
    for p in feature_target_files:
        print(f"  - {p.name}")

    dfs = []

    # 3.3 Dateien laden, Symbol-Spalte hinzufügen, auf relevante Spalten reduzieren
    for path in feature_target_files:
        symbol = path.stem.replace("_features_with_targets", "").upper()
        print(f"\n[INFO] Lade {path.name} (Symbol = {symbol})")

        df = pd.read_parquet(path)

        # Sicherstellen, dass wir einen DatetimeIndex haben
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index, utc=True)

        df = df.sort_index()
        df["symbol"] = symbol  # Symbol als Feature/Info behalten

        # Nur Spalten behalten, die wir wirklich fürs Modell wollen
        wanted_cols = ["symbol"]
        for c in FEATURE_COLS + TARGET_COLS:
            if c in df.columns:
                wanted_cols.append(c)
            else:
                print(f"[WARN] Spalte '{c}' nicht in {path.name} gefunden – wird ignoriert.")

        df = df[wanted_cols]
        dfs.append(df)

    # 3.4 Alle Symbole zu einem großen DataFrame zusammenführen
    full_df = pd.concat(dfs, axis=0).sort_index()
    print(
        f"\n[INFO] Gesamtdatensatz nach Concatenation: "
        f"{len(full_df):,} Zeilen, {len(full_df.columns)} Spalten."
    )

    # 3.5 Zeilen ohne Targets entfernen
    valid_target_cols = [c for c in TARGET_COLS if c in full_df.columns]
    before = len(full_df)
    full_df = full_df.dropna(subset=valid_target_cols)
    after = len(full_df)
    print(
        f"[INFO] Drop NaNs in Targets: {before:,} -> {after:,} Zeilen "
        f"({before - after:,} Zeilen entfernt)."
    )

    # 3.6 Train / Validation / Test anhand Zeitstempel splitten
    if not isinstance(full_df.index, pd.DatetimeIndex):
        full_df.index = pd.to_datetime(full_df.index, utc=True)

    train = full_df[full_df.index <= train_date]
    validation = full_df[
        (full_df.index > train_date) & (full_df.index <= val_date)
    ]
    test = full_df[
        (full_df.index > val_date) & (full_df.index <= test_date)
    ]

    # 3.7 Splits speichern
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(splits_dir / "usw_train.parquet", index=True)
    validation.to_parquet(splits_dir / "usw_validation.parquet", index=True)
    test.to_parquet(splits_dir / "usw_test.parquet", index=True)

    print("\n" + "=" * 70)
    print("Train/Validation/Test-Splits erstellt (nur AMZN, AAPL, META, NVDA, TSLA).")
    print(f"Train-Split:      {len(train):,} Zeilen")
    print(f"Validation-Split: {len(validation):,} Zeilen")
    print(f"Test-Split:       {len(test):,} Zeilen")
    print(f"Gespeichert unter: {splits_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
