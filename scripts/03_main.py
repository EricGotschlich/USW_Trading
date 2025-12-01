# scripts/main.py (oder scripts/03_main.py)

from pathlib import Path
import pandas as pd

from stock_feature_builder import FeatureBuilder, FeatureBuilderConfig


def load_price_parquet(prices_dir: Path, symbol: str) -> pd.DataFrame:
    """
    Lädt 1-Minuten-Preisdaten für ein Symbol aus einem Parquet:
    Erwartete Spalten: timestamp, open, high, low, close, volume, vwap, trade_count (optional)
    """
    path = prices_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Price file not found for {symbol}: {path}")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_data_prep_dates(project_root: Path):
    """
    Liest TRAIN_DATE, VALIDATION_DATE, TEST_DATE aus conf/params.yaml (Block DATA_PREP).
    """
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


def main():
    # ------------------------------------------------------------------
    # Pfade
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    prices_dir = data_dir / "raw" / "Prices_1m_adj"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    symbols_csv = data_dir / "nasdaq_100.csv"
    tickers_df = pd.read_csv(symbols_csv, encoding="latin1")
    all_tickers = sorted(tickers_df["Symbol"].dropna().unique().tolist())
    print(f"[INFO] {len(all_tickers)} Symbole aus {symbols_csv} geladen.")

    # ------------------------------------------------------------------
    # Index (QQQ) laden für Index-Features
    # ------------------------------------------------------------------
    index_symbol = "QQQ"
    index_df = load_price_parquet(prices_dir, index_symbol)
    print(
        f"[INFO] Index-Daten für {index_symbol} geladen: "
        f"{len(index_df):,} Zeilen."
    )

    # ------------------------------------------------------------------
    # Nur diese 5 Symbole verarbeiten
    # ------------------------------------------------------------------
    tickers_to_process = ["AMZN", "AAPL", "META", "NVDA", "TSLA"]
    print(
        f"[INFO] Starte Feature-Build ...\n"
        f"Verwende folgende Symbole für Feature-Build:\n{tickers_to_process}"
    )

    # ------------------------------------------------------------------
    # FeatureBuilder-Konfiguration
    # ------------------------------------------------------------------
    config = FeatureBuilderConfig(
        ema_windows=[15, 30, 60, 120],
        realized_vol_windows=[15, 30, 60, 120],
        volume_zscore_window=60,   # Volume-Z-Score über letzte 60 Minuten
        news_decay_lambda=0.0075,  # ~ 90 Minuten Halbwertszeit
    )

    # ------------------------------------------------------------------
    # Hauptschleife: alle Symbole durchgehen
    # ------------------------------------------------------------------
    for i, symbol in enumerate(tickers_to_process, start=1):
        print("\n" + "-" * 70)
        print(f"[{i}/{len(tickers_to_process)}] Verarbeite Symbol: {symbol}")
        print("-" * 70)

        if symbol.upper() == index_symbol.upper():
            print(f"[INFO] {symbol} ist Indexsymbol – wird hier übersprungen.")
            continue

        # Preisdaten laden
        try:
            price_df = load_price_parquet(prices_dir, symbol)
        except FileNotFoundError as e:
            print(f"[WARN] {e} – Symbol wird übersprungen.")
            continue

        if price_df.empty:
            print(f"[WARN] Leerer DataFrame für {symbol} – Symbol wird übersprungen.")
            continue

        # FeatureBuilder instanziieren
        fb = FeatureBuilder(
            df=price_df,
            symbol=symbol,
            project_root=project_root,
            index_df=index_df,
            index_price_col="close",
            timestamp_col="timestamp",
            config=config,
        )

        # Features bauen & speichern
        features_df = fb.build_features_before_split(
            save=True,
            save_suffix="",
        )

        print(
            f"[OK] Features für {symbol} erstellt: "
            f"{len(features_df):,} Zeilen, {len(features_df.columns)} Spalten."
        )

    print("\n" + "=" * 70)
    print("Feature-Build für alle Symbole abgeschlossen.")
    print(f"Ausgabepfade: {processed_dir} (pro Symbol *_features.parquet/.csv)")
    print("=" * 70 + "\n")

# ------------------------------------------------------------------
    # 3) Train / Validation / Test nach Datum splitten
    # ------------------------------------------------------------------
    train_date, val_date, test_date = load_data_prep_dates(project_root)
    print(
        f"[INFO] Data-Prep-Grenzen:\n"
        f"  TRAIN_DATE      = {train_date}\n"
        f"  VALIDATION_DATE = {val_date}\n"
        f"  TEST_DATE       = {test_date}"
    )

    # Indizes als DatetimeIndex sicherstellen
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index, utc=True)

    train = features_df[features_df.index <= train_date]
    validation = features_df[
        (features_df.index > train_date) & (features_df.index <= val_date)
    ]
    test = features_df[
        (features_df.index > val_date) & (features_df.index <= test_date)
    ]

    # Speicherort für Splits (z.B. data/processed/splits)
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(splits_dir / "usw_train.parquet", index=True)
    validation.to_parquet(splits_dir / "usw_validation.parquet", index=True)
    test.to_parquet(splits_dir / "usw_test.parquet", index=True)

    print("\n" + "=" * 70)
    print("Train/Validation/Test-Splits erstellt.")
    print(f"Train-Split:      {len(train):,} Zeilen")
    print(f"Validation-Split: {len(validation):,} Zeilen")
    print(f"Test-Split:       {len(test):,} Zeilen")
    print(f"Gespeichert unter: {splits_dir}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
