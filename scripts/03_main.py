# scripts/main.py (oder scripts/03_main.py)

from pathlib import Path
import pandas as pd
import yaml

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
        ema_windows=[5, 15],
        realized_vol_windows=[5, 10, 15],
        volume_zscore_window=15,
        news_decay_lambda=0.1386,
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


if __name__ == "__main__":
    main()
