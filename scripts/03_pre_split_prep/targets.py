# scripts/targets.py

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


class TargetBuilder:
    """
    Berechnet zukünftige VWAP-basierte Return-Targets für mehrere Zeithorizonte.

    Für jede Minute t:
      - berechnet VWAP über die nächsten N Minuten
      - Target = (VWAP_future - Current_Price) / Current_Price * 100
    """

    def __init__(self, df: pd.DataFrame, price_col: str = "vwap") -> None:
        self.df = df.copy()

        # Sicherstellen, dass wir einen DatetimeIndex haben
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "timestamp" in self.df.columns:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
                self.df = self.df.set_index("timestamp")
            else:
                self.df.index = pd.to_datetime(self.df.index, utc=True)

        self.df = self.df.sort_index()

        # Preis-Spalte bestimmen (Vorrang: vwap, sonst close)
        self.price_col = price_col
        if self.price_col not in self.df.columns:
            self.price_col = "close"
            if self.price_col not in self.df.columns:
                raise ValueError("Weder 'vwap' noch 'close' im DataFrame gefunden.")

        if "volume" not in self.df.columns:
            raise ValueError("Spalte 'volume' wird für VWAP-Targets benötigt.")

    def calculate_return_targets(self, windows: List[int]) -> pd.DataFrame:
        """
        Berechnet für jedes Fenster in 'windows' die zukünftigen
        VWAP-basierten Return-Targets in Prozent.

        Beispiel-Spalten:
          target_return_15m, target_return_30m, ...
        """
        print(f"[INFO] Berechne VWAP-Targets für Fenster (Minuten): {windows}")

        # Price * Volume vorbereiten
        self.df["pv"] = self.df[self.price_col] * self.df["volume"]

        # Für forward-looking Rolling-Window: DataFrame umdrehen
        df_rev = self.df.iloc[::-1]

        for w in windows:
            # Rolling-Summen über die nächsten w Minuten (im gespiegelten DF)
            roll_pv = df_rev["pv"].rolling(window=w).sum()
            roll_vol = df_rev["volume"].rolling(window=w).sum()

            # Wieder zurück spiegeln
            roll_pv = roll_pv.iloc[::-1]
            roll_vol = roll_vol.iloc[::-1]

            # Zukünftiger VWAP
            vwap_future = roll_pv / roll_vol

            # Eine Zeile nach oben schieben, damit t die Zukunft (t+1...t+w) bekommt
            vwap_future = vwap_future.shift(-1)

            current_price = self.df[self.price_col]
            target_col = f"target_return_{w}m"

            # Prozent-Return in %
            self.df[target_col] = (vwap_future - current_price) / current_price * 100.0

        # Hilfsspalte wieder entfernen
        self.df.drop(columns=["pv"], inplace=True)

        return self.df


def main():
    # ------------------------------------------------------------------
    # Pfade
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    # Wir nehmen alle *_features.parquet-Dateien
    feature_files = sorted(processed_dir.glob("*_features.parquet"))
    if not feature_files:
        print(f"[WARN] Keine Feature-Dateien in {processed_dir} gefunden.")
        return

    windows = [1, 5, 10, 15]

    print(f"[INFO] Found {len(feature_files)} Feature-Dateien.")
    for path in feature_files:
        symbol = path.stem.replace("_features", "")
        print("\n" + "-" * 70)
        print(f"[INFO] Berechne Targets für {symbol} ({path.name})")
        print("-" * 70)

        df = pd.read_parquet(path)

        tb = TargetBuilder(df)
        df_with_targets = tb.calculate_return_targets(windows)

        # >>> NEU: nur Zeilen behalten, in denen es in den letzten 15 Min News gab
        if "news_age_minutes" in df_with_targets.columns:
            before = len(df_with_targets)
            df_with_targets = df_with_targets[
                df_with_targets["news_age_minutes"].notna()
                & (df_with_targets["news_age_minutes"] <= 15)
                ]
            after = len(df_with_targets)
            print(
                f"[INFO] Filter news_age_minutes <= 15min: "
                f"{before:,} -> {after:,} Zeilen (für {symbol})"
            )
        else:
            print(f"[WARN] news_age_minutes nicht in {symbol}-DataFrame gefunden.")

        # Neue Dateien mit Targets (Features + Targets)
        out_parquet = processed_dir / f"{symbol}_features_with_targets.parquet"

        df_with_targets.to_parquet(out_parquet, index=True)

        print(
            f"[OK] Features + Targets gespeichert unter:\n"
            f"  {out_parquet}\n"
        )

    print("\n" + "=" * 70)
    print("Target-Berechnung für alle Symbole abgeschlossen.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
