# scripts/feature_plot.py
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Pfade & Setup
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # falls Datei in scripts/ liegt
DATA_DIR = PROJECT_ROOT / "data" / "processed"
IMG_DIR = PROJECT_ROOT / "images" / "data_preparation"
IMG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8")


def load_symbol_df(symbol: str) -> pd.DataFrame:
    """
    Lädt Features (+ Targets, falls vorhanden) für ein Symbol.
    Erwartet Dateien:
      data/processed/<SYMBOL>_features_with_targets.parquet
    oder fallback:
      data/processed/<SYMBOL>_features.parquet
    """
    symbol = symbol.upper()

    # Erst versuchen wir *_features_with_targets.parquet
    path = DATA_DIR / f"{symbol}_features_with_targets.parquet"
    if not path.exists():
        path = DATA_DIR / f"{symbol}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Keine Feature-Datei für {symbol} gefunden unter {path}")

    df = pd.read_parquet(path)

    # Index sicherstellen
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    return df


# -------------------------------------------------------------------
# 1) EMA(15) vs EMA(60) und EMA-Differenz
# -------------------------------------------------------------------
def plot_ema_and_diff(df: pd.DataFrame, symbol: str, n_points: int = 300) -> None:
    """
    Plot:
      - Close vs EMA(15) vs EMA(60)
      - (optional) EMA-Differenz: ema_diff_15_60

    Nur die letzten n_points Minuten, damit es lesbar bleibt.
    """
    symbol = symbol.upper()
    subset = df.tail(n_points).copy()

    # --- Plot 1: Close + EMAs ---
    plt.figure(figsize=(12, 6))
    plt.plot(subset.index, subset["close"], label="Close", alpha=0.6)
    if "ema_15" in subset.columns:
        plt.plot(subset.index, subset["ema_15"], label="EMA 15", linewidth=1.5)
    if "ema_60" in subset.columns:
        plt.plot(subset.index, subset["ema_60"], label="EMA 60", linewidth=1.5)

    plt.title(f"{symbol}: Close vs EMA(15) vs EMA(60) (letzte {n_points} Minuten)")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{symbol}_ema_15_60.png")
    plt.close()


# -------------------------------------------------------------------
# 2) Scatter: Aktie vs Index (z.B. 60m-Returns)
# -------------------------------------------------------------------
def plot_scatter_stock_vs_index(
    df: pd.DataFrame,
    symbol: str,
    window: int = 60,
) -> None:
    """
    Scatter-Plot: t-Minuten-Return der Aktie vs. t-Minuten-Return des Index (QQQ)
    mit Regressionslinie.

    Skala und Outlier-Filter werden abhängig vom Fenster gewählt.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    symbol = symbol.upper()
    stock_col = f"log_ret_{window}m"
    index_col = f"index_log_ret_{window}m"

    if stock_col not in df.columns or index_col not in df.columns:
        print(f"[WARN] {stock_col} oder {index_col} fehlen – Scatter wird übersprungen.")
        return

    # ---------- fensterabhängige Skala ----------
    if window == 60:
        limit = 0.05      # ±5 % einblenden
        axis = 0.06       # Achsen ±6 %
        tick_step = 0.01  # 1 %-Schritte
    elif window == 15:
        limit = 0.03
        axis = 0.035
        tick_step = 0.005
    elif window == 120:
        limit = 0.07
        axis = 0.08
        tick_step = 0.01
    else:
        # Fallback
        limit = 0.04
        axis = 0.05
        tick_step = 0.01

    # Daten auswählen und NaNs entfernen
    data = df[[stock_col, index_col]].dropna().rename(
        columns={stock_col: "stock_ret", index_col: "index_ret"}
    )

    # Ausreißer filtern, damit die Wolke sichtbar bleibt
    mask = data["stock_ret"].between(-limit, limit) & data["index_ret"].between(
        -limit, limit
    )
    data = data[mask]

    if data.empty:
        print("[WARN] Keine Daten nach Outlier-Filter.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8), facecolor="white")

    sns.regplot(
        data=data,
        x="stock_ret",
        y="index_ret",
        scatter_kws={"alpha": 0.2, "s": 18},
        line_kws={"color": "red", "linewidth": 2},
    )

    # Achsen-Ticks je nach Fenster
    ticks = np.arange(-axis, axis + tick_step / 2, tick_step)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(-axis, axis)
    plt.ylim(-axis, axis)

    plt.title(
        f"Scatter: {window}-Min Returns ({symbol} vs QQQ) with Regression Line",
        fontsize=14,
    )
    plt.xlabel(f"{symbol} {window}-Min Return (log_ret_{window}m)", fontsize=12)
    plt.ylabel(f"QQQ {window}-Min Return (index_log_ret_{window}m)", fontsize=12)

    plt.tight_layout()
    out_path = IMG_DIR / f"{symbol}_scatter_vs_index_{window}m.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Scatter-Plot gespeichert unter: {out_path}")


# -------------------------------------------------------------------
# 3) Intraday-Returns (ähnlich wie bei der anderen Gruppe)
# -------------------------------------------------------------------
def plot_intraday_log_returns(
    df: pd.DataFrame,
    symbol: str,
    day: str = "2025-11-20",
) -> None:
    """
    Intraday-Plot der stündlichen Mittelwerte von
    log_ret_1m, log_ret_15m, log_ret_60m für einen Tag.

    Struktur ist absichtlich sehr ähnlich zum Beispiel mit
    simple_return_1m/5m/15m – nur eben mit deinen Log-Return-Features.
    """
    symbol = symbol.upper()

    # Wir arbeiten mit den Log-Return-Spalten aus deinen Features
    cols = ["log_ret_1m", "log_ret_15m", "log_ret_60m"]
    cols = [c for c in cols if c in df.columns]

    if not cols:
        print("[WARN] Keine passenden log_ret_* Spalten gefunden – Intraday-Plot wird übersprungen.")
        return

    # Tag auswählen
    day_dt = pd.to_datetime(day).date()
    df_day = df[df.index.date == day_dt]

    if df_day.empty:
        available_days = sorted(set(df.index.date))
        print(
            f"[WARN] Keine Daten für {day_dt}! "
            f"Verfügbare Tage: {available_days[:5]} ... (insgesamt {len(available_days)})"
        )
        return

    # Zeitfilter wie im Beispiel: 06:00–22:00 (UTC)
    df_day = df_day[
        (df_day.index.hour >= 6) &
        (df_day.index.hour <= 22)
    ]

    if df_day.empty:
        print(f"[WARN] Nach Zeitfilter (06–22 Uhr) keine Daten mehr für {day_dt}.")
        return

    # Stündlich aggregieren (Mittelwerte der Log-Returns)
    df_hourly = df_day[cols].resample("1H").mean()

    # Plot
    plt.figure(figsize=(16, 8))
    for c in cols:
        plt.plot(df_hourly.index, df_hourly[c], label=c)

    plt.title(f"{symbol}: Stündliche Log-Returns für {day} (06:00–22:00)")
    plt.xlabel("Uhrzeit")
    plt.ylabel("Log-Returns (Mittelwert pro Stunde)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = IMG_DIR / f"{symbol}_intraday_log_returns_{day}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Intraday-Returns-Plot gespeichert unter: {out_path}")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
def main():
    SYMBOL = "AAPL"

    print(f"[INFO] Lade Daten für {SYMBOL} ...")
    df = load_symbol_df(SYMBOL)

    print("[INFO] Erzeuge EMA-Plots ...")
    plot_ema_and_diff(df, SYMBOL, n_points=300)

    print("[INFO] Erzeuge Scatter-Plot Stock vs Index ...")
    plot_scatter_stock_vs_index(
        df,
        SYMBOL,
        window=60,  # 60-Minuten-Returns
    )

    print("[INFO] Erzeuge Intraday-Returns-Plot (Log-Returns) ...")
    # Datum kannst du frei wählen, nimm einen Tag wo du sicher Daten hast
    plot_intraday_log_returns(
        df,
        SYMBOL,
        day="2025-11-20",
    )

    print(f"[OK] Plots gespeichert unter: {IMG_DIR}")



if __name__ == "__main__":
    main()
