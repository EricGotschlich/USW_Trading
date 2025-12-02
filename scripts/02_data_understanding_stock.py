"""
Data Understanding für Einzelaktien (NASDAQ-100 Beispiel: AAPL)

Erzeugt pro Symbol folgende Plots unter <PROJECT_ROOT>/images:

1) stock_<SYMBOL>_moving_averages.png
   -> täglicher Schlusskurs + 20/50/200-Tage-MA

2) stock_<SYMBOL>_volume_over_time.png
   -> tägliches Handelsvolumen

3) stock_<SYMBOL>_volatility_30d.png
   -> 30-Tage-Rolling-Volatilität der Tages-Log-Renditen

4) stock_<SYMBOL>_logret_1m_hist.png
   -> Histogramm der 1-Minuten-Log-Returns (geclipped auf ±1 %)

5) stock_<SYMBOL>_vs_QQQ_daily_returns.png (optional)
   -> Scatterplot: tägliche Rendite der Aktie vs. tägliche QQQ-Rendite,
      falls QQQ-Daten unter data/raw/Prices_1m_adj/QQQ.parquet vorhanden sind.

Voraussetzung:
- data/raw/Prices_1m_adj/<SYMBOL>.parquet   (1-Minuten-Bars aus deinem Data-Acquisition-Script)
- optional: data/raw/Prices_1m_adj/QQQ.parquet für den Korrelations-Plot
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# etwas hübschere Standard-Optik
plt.style.use("seaborn-v0_8-whitegrid")


# -------------------------------------------------------------
# Helper: Preise für ein Symbol laden
# -------------------------------------------------------------
def load_minute_data(prices_dir: Path, symbol: str) -> pd.DataFrame:
    path = prices_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Preis-Datei nicht gefunden: {path}")

    df = pd.read_parquet(path)

    # Erwartete Spalten: timestamp, open, high, low, close, volume, ...
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp'-Spalte fehlt in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    return df


def main():
    # ---------------------------------------------------------
    # Pfade
    # ---------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    prices_dir = project_root / "data" / "raw" / "Prices_1m_adj"
    images_dir = project_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Welche Aktien anschauen?
    SYMBOLS = ["AAPL"]  # hier kannst du z.B. ["AAPL", "NVDA"] eintragen
    INDEX_SYMBOL = "QQQ"
    qqq_path = prices_dir / f"{INDEX_SYMBOL}.parquet"

    # QQQ optional für Korrelations-Plot laden
    df_qqq_daily = None
    if qqq_path.exists():
        df_qqq_min = load_minute_data(prices_dir, INDEX_SYMBOL)
        df_qqq_daily = (
            df_qqq_min
            .resample("1D")
            .agg({"close": "last"})
            .dropna(how="any")
        )
        df_qqq_daily["ret_qqq"] = np.log(df_qqq_daily["close"]).diff()

    # ---------------------------------------------------------
    # Schleife über alle gewünschten Aktien
    # ---------------------------------------------------------
    for symbol in SYMBOLS:
        print(f"\n=== Data Understanding für {symbol} ===")

        df_min = load_minute_data(prices_dir, symbol)

        # 1-Minuten-Log-Returns für spätere Histogramme
        df_min["log_ret_1m"] = np.log(df_min["close"]).diff()

        print(f"{symbol} – erste Zeilen (1-Minuten-Daten):")
        print(df_min.head())
        print("\nInfo (1-Minuten-Daten):")
        print(df_min[["open", "close", "volume"]].info())
        print("\nDeskriptive Statistik (open, close, volume):")
        print(df_min[["open", "close", "volume"]].describe())

        # -----------------------------------------------------
        # Auf Tagesebene aggregieren
        # -----------------------------------------------------
        daily = df_min.resample("1D").agg(
            {
                "open": "first",   # erster Kurs des Tages
                "close": "last",   # letzter Kurs des Tages
                "volume": "sum",   # Tagesvolumen
            }
        )
        daily = daily.dropna(how="any")  # nur Handelstage behalten

        # 20- und 200-Tage MAs auf Schlusskurs
        daily["ma20"] = daily["close"].rolling(window=20, min_periods=1).mean()
        daily["ma200"] = daily["close"].rolling(window=200, min_periods=1).mean()

        # Tägliche Log-Renditen
        daily["ret_stock"] = np.log(daily["close"]).diff()

        # -----------------------------------------------------
        # Plot 1: Schlusskurs + 20/200-Tage-MA
        # -----------------------------------------------------
        plt.figure(figsize=(12, 5))
        plt.plot(daily.index, daily["close"], label="Täglicher Schlusskurs", linewidth=0.8)
        plt.plot(daily.index, daily["ma20"], label="20-Tage MA", linewidth=2)
        plt.plot(daily.index, daily["ma200"], label="200-Tage MA", linewidth=2)

        plt.title(f"{symbol}: Schlusskurs mit 20/200-Tage gleitendem Durchschnitt")
        plt.xlabel("Datum")
        plt.ylabel("Preis in USD")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = images_dir / f"stock_{symbol}_moving_averages.png"
        plt.savefig(out_path)
        print(f"[OK] Saved: {out_path}")
        plt.close()

        # -----------------------------------------------------
        # Plot 2: tägliches Handelsvolumen
        # -----------------------------------------------------
        plt.figure(figsize=(12, 3.5))
        plt.plot(daily.index, daily["volume"])
        plt.title(f"{symbol}: tägliches Handelsvolumen")
        plt.xlabel("Datum")
        plt.ylabel("Volumen (Shares)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = images_dir / f"stock_{symbol}_volume_over_time.png"
        plt.savefig(out_path)
        print(f"[OK] Saved: {out_path}")
        plt.close()

        # -----------------------------------------------------
        # Plot 3: 30-Tage-Rolling-Volatilität (tägliche Log-Renditen)
        # -----------------------------------------------------
        vol_30d = daily["ret_stock"].rolling(window=30, min_periods=5).std() * 100.0

        plt.figure(figsize=(12, 5))
        plt.plot(vol_30d.index, vol_30d.values)
        plt.title(f"{symbol}: 30-Tage-Rolling-Volatilität (tägliche Log-Renditen)")
        plt.xlabel("Datum")
        plt.ylabel("Volatilität in %")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = images_dir / f"stock_{symbol}_volatility_30d.png"
        plt.savefig(out_path)
        print(f"[OK] Saved: {out_path}")
        plt.close()

        # -----------------------------------------------------
        # Plot 4: Histogramme der t-Minuten-Log-Returns in EINER PNG
        #        für t = [15, 30, 60, 120]
        # -----------------------------------------------------
        horizons = [15, 30, 60]

        # t-Minuten-Returns vorbereiten
        for t in horizons:
            col_name = f"log_ret_{t}m"
            df_min[col_name] = np.log(df_min["close"].shift(-t)) - np.log(df_min["close"])

        # Figure mit Subplots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for ax, t in zip(axes, horizons):
            col_name = f"log_ret_{t}m"
            returns_t = df_min[col_name].dropna()

            # Clip, damit extreme Ausreißer die Skala nicht zerstören
            returns_t_clip = returns_t.clip(lower=-0.05, upper=0.05)

            ax.hist(returns_t_clip, bins=80, density=True)
            ax.set_title(f"{symbol}: {t}-Minuten-Log-Returns (±5 %)")
            ax.set_xlabel(f"{t}-Minuten-Log-Return")
            ax.set_ylabel("Dichte")

        plt.suptitle(f"{symbol}: Verteilung der t-Minuten-Log-Returns", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = images_dir / f"stock_{symbol}_logret_multi_horizons.png"
        plt.savefig(out_path)
        print(f"[OK] Saved: {out_path}")
        plt.close()






    print("\nFertig. Alle Stock-Plots unter:", images_dir)


if __name__ == "__main__":
    main()
