"""
Data Understanding für Index-Features (QQQ als NASDAQ-100 Proxy)

Erzeugt vier Plots und speichert sie unter <PROJECT_ROOT>/images:

1) index_moving_averages.png   -> täglicher Schlusskurs + 20/50/200-Tage-MA
2) index_volume_over_time.png  -> tägliches Handelsvolumen
3) index_weekly_close.png      -> wöchentlicher Schlusskurs
4) index_volatility_30d.png    -> 30-Tage-Rolling-Volatilität der Tagesrenditen

Voraussetzung:
- data/raw/Prices_1m_adj/QQQ.parquet  (1-Minuten-Bars aus deinem Data-Acquisition-Script)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional etwas hübschere Standard-Optik
plt.style.use("seaborn-v0_8-whitegrid")


def main():
    # -------------------------------------------------------------
    # Pfade
    # -------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent          # .../USW_Trading/scripts
    project_root = script_dir.parent                      # .../USW_Trading
    prices_dir = project_root / "data" / "raw" / "Prices_1m_adj"
    images_dir = project_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    index_symbol = "QQQ"
    price_path = prices_dir / f"{index_symbol}.parquet"

    print(f"PROJECT_ROOT: {project_root}")
    print(f"Erwarte Datei: {price_path}")

    if not price_path.exists():
        raise FileNotFoundError(f"Preis-Datei nicht gefunden: {price_path}")

    # -------------------------------------------------------------
    # 1. Daten laden & Basis-Checks
    # -------------------------------------------------------------
    df = pd.read_parquet(price_path)

    # Erwartete Spalten: timestamp, open, high, low, close, volume, ...
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    print(f"{index_symbol} – erste Zeilen:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nDeskriptive Statistik (open, close, volume):")
    print(df[["open", "close", "volume"]].describe())

    # -------------------------------------------------------------
    # 2. Auf Tagesebene aggregieren
    #    -> nur Handelstage, Wochenenden/Feiertage werden zu NaN und dann gedroppt
    # -------------------------------------------------------------
    daily = df.resample("1D").agg(
        {
            "open": "first",   # erster Kurs des Tages
            "close": "last",   # letzter Kurs des Tages
            "volume": "sum",   # Tagesvolumen
        }
    )

    daily = daily.dropna(how="any")  # entfernt Tage ohne Handel

    daily["ma20"] = daily["close"].rolling(window=20, min_periods=1).mean()
    daily["ma50"] = daily["close"].rolling(window=50, min_periods=1).mean()
    daily["ma200"] = daily["close"].rolling(window=200, min_periods=1).mean()

    # -------------------------------------------------------------
    # 3. Plot: Schlusskurs + 20/200-Tage Moving Average
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily["close"], label="Täglicher Schlusskurs", linewidth=0.8)
    plt.plot(daily.index, daily["ma20"], label="20-Tage MA", linewidth=2)
    plt.plot(daily.index, daily["ma200"], label="200-Tage MA", linewidth=2)

    plt.title(f"{index_symbol}: Schlusskurs mit 20/200-Tage gleitendem Durchschnitt")
    plt.xlabel("Datum")
    plt.ylabel("Preis in USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = images_dir / "index_moving_averages.png"
    plt.savefig(out_path)
    print(f"[OK] Saved: {out_path}")



    # -------------------------------------------------------------
    # 5. Plot: wöchentlicher Schlusskurs (z.B. Freitags)
    # -------------------------------------------------------------
    weekly_close = daily["close"].resample("W-FRI").last()

    plt.figure(figsize=(12, 5))
    plt.plot(weekly_close.index, weekly_close.values, label="Wöchentlicher Schlusskurs")
    plt.title(f"{index_symbol}: wöchentlicher Schlusskurs (Freitag)")
    plt.xlabel("Datum")
    plt.ylabel("Schlusskurs in USD")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = images_dir / "index_weekly_close.png"
    plt.savefig(out_path)
    print(f"[OK] Saved: {out_path}")

    # -------------------------------------------------------------
    # 6. Plot: 30-Tage-Rolling-Volatilität
    #    Volatilität = Std-Abw. der täglichen Log-Renditen * 100 (in %)
    # -------------------------------------------------------------
    daily_returns = np.log(daily["close"]).diff()
    vol_30d = daily_returns.rolling(window=30, min_periods=5).std() * 100.0

    plt.figure(figsize=(12, 5))
    plt.plot(vol_30d.index, vol_30d.values)
    plt.title(f"{index_symbol}: 30-Tage-Rolling-Volatilität (tägliche Log-Renditen)")
    plt.xlabel("Datum")
    plt.ylabel("Volatilität in %")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = images_dir / "index_volatility_30d.png"
    plt.savefig(out_path)
    print(f"[OK] Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
