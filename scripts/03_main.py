"""
Build-All-Features Script für NASDAQ-100 Projekt

- Lädt alle NASDAQ-100 Symbole aus der CSV (SYMBOLS_CSV in conf/params.yaml)
- Lädt 1-Minuten-Preisdaten aus data/raw/Prices_1m_adj/<SYMBOL>.parquet
- Lädt QQQ als Index und reicht ihn an den FeatureBuilder weiter
- Ruft für jedes Symbol den FeatureBuilder auf (inkl.:
  * Log-Returns
  * EMAs & EMA-Differenzen
  * Z-Score
  * Realisierte Volatilität
  * High-Low-Span
  * avg_volume_per_trade
  * News-Alignment mit Exponential Decay
  * Index-Features auf Basis QQQ)
- Speichert die fertigen Features nach data/processed/<SYMBOL>_features.parquet/.csv
"""

import os
from pathlib import Path

import yaml
import pandas as pd

from stock_feature_builder import FeatureBuilder


def load_config_and_paths():
    """
    Lädt params.yaml und leitet die relevanten Pfade ab.
    Erwartet:
      conf/params.yaml mit Block DATA_ACQUISITION:
        DATA_PATH: "data/raw"
        SYMBOLS_CSV: "conf/nasdaq_100.csv" (oder ähnlicher relativer Pfad)
    """
    project_root = Path(__file__).resolve().parents[1]
    conf_path = project_root / "conf" / "params.yaml"

    if not conf_path.exists():
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_cfg = params["DATA_ACQUISITION"]

    # Rohdaten-Basis (z.B. "data/raw")
    raw_base = Path(data_cfg["DATA_PATH"])
    if not raw_base.is_absolute():
        raw_base = (project_root / raw_base).resolve()

    prices_dir = raw_base / "Prices_1m_adj"

    # Pfad zur NASDAQ-100 Symbol-Liste
    symbols_csv = Path(data_cfg["SYMBOLS_CSV"])
    if not symbols_csv.is_absolute():
        symbols_csv = (project_root / symbols_csv).resolve()

    processed_dir = project_root / "data" / "processed"

    return project_root, prices_dir, symbols_csv, processed_dir


def load_symbol_list(symbols_csv: Path) -> list[str]:
    """
    Lädt die Liste der NASDAQ-100 Symbole aus der CSV.
    Erwartet eine Spalte "Symbol".
    """
    df = pd.read_csv(symbols_csv, encoding="latin1", engine="python")
    tickers = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    print(f"[INFO] {len(tickers)} Symbole aus {symbols_csv} geladen.")
    return tickers


def load_price_parquet(prices_dir: Path, symbol: str) -> pd.DataFrame:
    """
    Lädt eine Parquet-Datei für ein Symbol aus dem Prices_1m_adj-Ordner.
    Erwartet Spalten: timestamp, open, high, low, close, volume, ...
    """
    path = prices_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Preisdatei für {symbol} nicht gefunden: {path}")

    df = pd.read_parquet(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def main():
    # ------------------------------------------------------------
    # Pfade & Konfiguration laden
    # ------------------------------------------------------------
    project_root, prices_dir, symbols_csv, processed_dir = load_config_and_paths()
    processed_dir.mkdir(parents=True, exist_ok=True)

    # NASDAQ-100 Symbol-Liste
    tickers = load_symbol_list(symbols_csv)

    # ------------------------------------------------------------
    # Index-Daten (QQQ) laden für Index-Features
    # ------------------------------------------------------------
    index_symbol = "QQQ"
    try:
        index_df = load_price_parquet(prices_dir, index_symbol)
        print(f"[INFO] Index-Daten für {index_symbol} geladen: {len(index_df):,} Zeilen.")
    except FileNotFoundError as e:
        # Wenn QQQ fehlt, kannst du Index-Features später im FeatureBuilder deaktivieren
        print(f"[WARN] {e}")
        index_df = None

    # ------------------------------------------------------------
    # Optionale Steuerung: ab welchem Symbol starten?
    # (falls du später nur einen Teil laufen lassen willst)
    # ------------------------------------------------------------
    START_AT_INDEX = 0      # z.B. 20, um erst beim 21. Symbol anzufangen
    MAX_SYMBOLS = None      # z.B. 10, um nur 10 Symbole zu bearbeiten

    tickers_to_process = tickers[START_AT_INDEX:]
    if MAX_SYMBOLS is not None:
        tickers_to_process = tickers_to_process[:MAX_SYMBOLS]

    print(
        f"[INFO] Starte Feature-Build für {len(tickers_to_process)} Symbole "
        f"(von insgesamt {len(tickers)}) ..."
    )

    # ------------------------------------------------------------
    # Ticker-Auswahl auf kleines Set einschränken
    # ------------------------------------------------------------
    focus_symbols = {"META", "TSLA", "AMZN", "AAPL", "NVDA"}

    # falls 'tickers' vorher aus der CSV kommt:
    tickers_to_process = [t for t in tickers if t.upper() in focus_symbols]

    print("Verwende folgende Symbole für Feature-Build:")
    print(tickers_to_process)

    # ------------------------------------------------------------
    # Hauptschleife: alle Symbole durchgehen
    # ------------------------------------------------------------
    for i, symbol in enumerate(tickers_to_process, start=1):
        print("\n" + "-" * 70)
        print(f"[{i}/{len(tickers_to_process)}] Verarbeite Symbol: {symbol}")
        print("-" * 70)

        # QQQ als "Aktie" überspringen, wenn du ihn nur als Index nutzen willst
        # (falls du QQQ auch als eigenes Training-Symbol willst, entferne den continue)
        if symbol.upper() == index_symbol.upper():
            print(f"[INFO] {symbol} ist Indexsymbol – wird in dieser Schleife übersprungen.")
            continue

        # Preisdaten für dieses Symbol laden
        try:
            price_df = load_price_parquet(prices_dir, symbol)
        except FileNotFoundError as e:
            print(f"[WARN] {e} – Symbol wird übersprungen.")
            continue

        if price_df.empty:
            print(f"[WARN] Leerer DataFrame für {symbol} – Symbol wird übersprungen.")
            continue

        # --------------------------------------------------------
        # FeatureBuilder instanziieren
        # --------------------------------------------------------
        fb = FeatureBuilder(
            df=price_df,
            symbol=symbol,
            ema_windows=[5, 15, 30, 60, 120],
            realized_vol_windows=[15, 30, 60],
            zscore_window=30,
            timestamp_col="timestamp",
            # Index-Features: falls index_df None ist, ignoriert der FeatureBuilder das
            index_df=index_df,
            index_price_col="close",
        )

        # --------------------------------------------------------
        # Features bauen & speichern
        # --------------------------------------------------------
        features_df = fb.build_features_before_split(
            save=True,
            save_suffix="",   # optional, z.B. "_v1" falls du Versionen willst
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
