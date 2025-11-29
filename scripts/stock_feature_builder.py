from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class FeatureBuilder:
    """
    Feature-Engineering für das NASDAQ-100-Projekt (Pre-Split).

    Erwartet einen DataFrame mit mindestens:
      - 'timestamp'
      - 'open', 'high', 'low', 'close', 'volume'
      - optional: 'trade_count'

    Optional:
      - News aus data/processed/nasdaq_news_with_sentiment.parquet
      - Index-Daten (z.B. QQQ) als index_df, um Index-Features + Relative Returns zu bauen.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        ema_windows: Optional[List[int]] = None,
        realized_vol_windows: Optional[List[int]] = None,
        zscore_window: int = 30,
        timestamp_col: str = "timestamp",
        index_df: Optional[pd.DataFrame] = None,
        index_price_col: str = "close",
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            1-Minuten-OHLCV-Daten einer Aktie.
        symbol : str
            Tickersymbol (z.B. 'AAPL') – wird genutzt, um passende News zu filtern.
        ema_windows : List[int]
            Zeitfenster in Minuten für EMAs (Close & Volume).
        realized_vol_windows : List[int]
            Zeitfenster in Minuten für realisierte Volatilität.
        zscore_window : int
            Fenstergröße (Minuten) für Z-Score des Close-Preises.
        timestamp_col : str
            Spaltenname mit Zeitstempeln.
        index_df : pd.DataFrame, optional
            1-Minuten-Daten des Index (z.B. QQQ), um Index-Features + Relative Returns zu bauen.
        index_price_col : str
            Spalte im index_df, die den Preis enthält (z.B. 'close').
        """
        self.symbol = symbol.upper()
        self.df = df.copy()

        # Default-Fenster passend zu deiner Problemdefinition
        self.ema_windows = ema_windows or [5, 15, 30, 60, 120]
        # Realisierte Volatilität z.B. für 15, 30, 60 Minuten (optional 120 kannst du ergänzen)
        self.realized_vol_windows = realized_vol_windows or [15, 30, 60]
        self.zscore_window = zscore_window
        self.timestamp_col = timestamp_col

        # Projekt-Root & Data-Verzeichnis bestimmen
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data"

        # Timestamp in Index bringen
        if self.timestamp_col in self.df.columns:
            self.df[self.timestamp_col] = pd.to_datetime(
                self.df[self.timestamp_col], utc=True
            )
            self.df = self.df.set_index(self.timestamp_col)

        # Sicherstellen, dass Index DatetimeIndex ist und sortiert
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index, utc=True)

        self.df = self.df.sort_index()

        # ---------------- Index (z.B. QQQ) vorbereiten ---------------- #
        self.index_df = None
        self.index_price_col = index_price_col

        if index_df is not None:
            idx = index_df.copy()
            if self.timestamp_col in idx.columns:
                idx[self.timestamp_col] = pd.to_datetime(
                    idx[self.timestamp_col], utc=True
                )
                idx = idx.set_index(self.timestamp_col)

            if not isinstance(idx.index, pd.DatetimeIndex):
                idx.index = pd.to_datetime(idx.index, utc=True)

            self.index_df = idx.sort_index()

    # ------------------------------------------------------------------
    # 1. 1-Minuten-Log-Returns und Vergangenheits-Returns
    # ------------------------------------------------------------------
    def _add_log_returns(self, past_return_windows: Optional[List[int]] = None) -> List[int]:
        """
        Fügt hinzu:
          - log_ret_1m: log(close_t) - log(close_{t-1})
          - log_ret_{w}m: Summe der 1m-Log-Returns über die letzten w Minuten

        Gibt die tatsächlich verwendeten Fenster zurück.
        """
        # Basis 1m-Log-Return
        self.df["log_ret_1m"] = np.log(self.df["close"]).diff()

        # Standard-Fenster: passend zu deiner Problemdefinition
        past_return_windows = past_return_windows or [15, 30, 60, 120]

        for w in past_return_windows:
            col_name = f"log_ret_{w}m"
            self.df[col_name] = (
                self.df["log_ret_1m"].rolling(window=w).sum()
            )

        return past_return_windows

    # ------------------------------------------------------------------
    # 2. EMAs für Close & Volume
    # ------------------------------------------------------------------
    def _add_emas(self) -> None:
        """
        Fügt EMAs für Close-Preis und Volumen hinzu:
          - ema_close_{w}m
          - ema_volume_{w}m
        für w in self.ema_windows.
        """
        for w in self.ema_windows:
            self.df[f"ema_close_{w}m"] = (
                self.df["close"].ewm(span=w, adjust=False).mean()
            )
            self.df[f"ema_volume_{w}m"] = (
                self.df["volume"].ewm(span=w, adjust=False).mean()
            )

    # ------------------------------------------------------------------
    # 3. EMA-Differenzen (Trend-Signal)
    # ------------------------------------------------------------------
    def _add_ema_differences(
        self,
        pairs: Optional[List[tuple[int, int]]] = None,
    ) -> None:
        """
        Fügt Differenzen zwischen schnellen und langsamen EMAs hinzu.
        Default:
          (5, 30)  -> ema_diff_close_5_30
          (15, 60) -> ema_diff_close_15_60
        """
        if pairs is None:
            pairs = [(5, 30), (15, 60)]

        for fast, slow in pairs:
            fast_col = f"ema_close_{fast}m"
            slow_col = f"ema_close_{slow}m"
            diff_col = f"ema_diff_close_{fast}_{slow}"

            if fast_col in self.df.columns and slow_col in self.df.columns:
                self.df[diff_col] = self.df[fast_col] - self.df[slow_col]
            else:
                print(
                    f"[INFO] Konnte {diff_col} nicht berechnen; "
                    f"{fast_col} oder {slow_col} fehlt."
                )

    # ------------------------------------------------------------------
    # 4. Z-Score des Close-Preises
    # ------------------------------------------------------------------
    def _add_zscore(self, window: Optional[int] = None) -> None:
        """
        Z-Score = (Close - RollingMean) / RollingStd
        Rolling-Fenster nur mit Vergangenheitsdaten (shift(1)).
        """
        w = window or self.zscore_window
        rolling_mean = self.df["close"].rolling(window=w).mean().shift(1)
        rolling_std = self.df["close"].rolling(window=w).std().shift(1)
        self.df[f"zscore_close_{w}m"] = (
            (self.df["close"] - rolling_mean) / rolling_std
        )

    # ------------------------------------------------------------------
    # 5. Realisierte Volatilität über mehrere Fenster
    # ------------------------------------------------------------------
    def _add_realized_volatility(self) -> None:
        """
        Realisierte Volatilität:
          rv_{W}m = sqrt( Sum_{i=t-W+1..t} (log_ret_1m_i)^2 )
        """
        if "log_ret_1m" not in self.df.columns:
            # Sicherheit, falls _add_log_returns noch nicht aufgerufen wurde
            self._add_log_returns(past_return_windows=[15, 30, 60, 120])

        for w in self.realized_vol_windows:
            col_name = f"rv_{w}m"
            self.df[col_name] = (
                self.df["log_ret_1m"]
                .rolling(window=w)
                .apply(lambda x: np.sqrt((x ** 2).sum()), raw=False)
            )

    # ------------------------------------------------------------------
    # 6. High-Low-Spanne pro Bar
    # ------------------------------------------------------------------
    def _add_high_low_span(self) -> None:
        """
        Fügt Spalte 'hl_span' hinzu: high - low pro 1-Minuten-Bar.
        """
        if "high" in self.df.columns and "low" in self.df.columns:
            self.df["hl_span"] = self.df["high"] - self.df["low"]
        else:
            print("[WARN] 'high' oder 'low' fehlt – hl_span nicht berechnet.")

    # ------------------------------------------------------------------
    # 7. Average Volume per Trade
    # ------------------------------------------------------------------
    def _add_avg_volume_per_trade(
        self,
        volume_col: str = "volume",
        trades_col: str = "trade_count",
    ) -> None:
        """
        avg_volume_per_trade = volume / trade_count
        """
        if volume_col in self.df.columns and trades_col in self.df.columns:
            self.df["avg_volume_per_trade"] = (
                self.df[volume_col] / self.df[trades_col].replace(0, np.nan)
            )
        else:
            print(
                f"[INFO] Columns '{volume_col}' oder '{trades_col}' fehlen – "
                "avg_volume_per_trade nicht berechnet."
            )

    # ------------------------------------------------------------------
    # 8. News-Ausrichtung mit Exponential Decay
    # ------------------------------------------------------------------
    def _align_news_with_price(self, lambda_decay: float = 0.0075) -> None:
        """
        Align News-Sentiment pro Aktie mit Preis-Daten.

        Vorgehen:
        - Lade data/processed/nasdaq_news_with_sentiment.parquet
        - Filtere auf self.symbol
        - Für jede 1-Minuten-Bar:
            * finde die letzte News vor/gleich diesem Zeitpunkt
            * news_age_minutes = (bar_time - news_time)
            * effective_sentiment = sentiment * exp(-lambda_decay * news_age_minutes)

        Ergebnis-Spalten:
        - last_news_sentiment
        - news_age_minutes
        - effective_sentiment_t
        - news_id
        - news_headline
        """
        news_path = self.data_dir / "processed" / "nasdaq_news_with_sentiment.parquet"

        if not news_path.exists():
            print(
                f"[WARN] News-Datei {news_path} nicht gefunden. "
                "News-Features werden nicht berechnet."
            )
            return

        news_df = pd.read_parquet(news_path)

        # Erwartete Spalten: symbol, created_at, sentiment_score, headline, id, ...
        if "symbol" not in news_df.columns:
            print(
                "[WARN] Spalte 'symbol' nicht in News-Parquet – "
                "kann nicht pro Aktie filtern. News werden ignoriert."
            )
            return

        if "sentiment_score" not in news_df.columns:
            print(
                "[WARN] Spalte 'sentiment_score' fehlt in News-Parquet – "
                "News werden ignoriert."
            )
            return

        # Zeitspalte bestimmen
        ts_col = None
        for candidate in ["created_at", "timestamp", "updated_at"]:
            if candidate in news_df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            print(
                "[WARN] Keine Zeitspalte ('created_at'/'timestamp'/'updated_at') "
                "im News-Parquet gefunden – News werden ignoriert."
            )
            return

        # Auf Symbol filtern
        news_symbol_df = news_df[news_df["symbol"].str.upper() == self.symbol].copy()
        if news_symbol_df.empty:
            print(
                f"[INFO] Keine News für Symbol {self.symbol} in {news_path} gefunden."
            )
            return

        news_symbol_df[ts_col] = pd.to_datetime(
            news_symbol_df[ts_col], errors="coerce", utc=True
        )
        news_symbol_df = news_symbol_df.dropna(subset=[ts_col])
        news_symbol_df = news_symbol_df.sort_values(ts_col).reset_index(drop=True)

        # Zielspalten im Preis-DF initialisieren
        self.df["last_news_sentiment"] = 0.0
        self.df["news_age_minutes"] = np.nan
        self.df["effective_sentiment_t"] = 0.0
        self.df["news_id"] = None
        self.df["news_headline"] = None

        news_times = news_symbol_df[ts_col].values
        news_sentiments = news_symbol_df["sentiment_score"].values
        news_ids = news_symbol_df["id"].astype(str).values if "id" in news_symbol_df.columns else [None] * len(news_symbol_df)
        news_headlines = news_symbol_df["headline"].astype(str).values if "headline" in news_symbol_df.columns else [None] * len(news_symbol_df)

        n_news = len(news_symbol_df)
        news_idx = 0
        total_rows = len(self.df)

        print(
            f"[INFO] Aligning {n_news} News-Items von {self.symbol} "
            f"mit {total_rows} Preis-Bars (λ={lambda_decay}) ..."
        )

        for i, (ts, _) in enumerate(self.df.iterrows()):
            current_time = ts.to_datetime64()

            # Pointer vorwärts bewegen, solange die nächste News <= current_time ist
            while (
                news_idx < n_news - 1
                and news_times[news_idx + 1] <= current_time
            ):
                news_idx += 1

            # Prüfen, ob es News vor/gleich current_time gibt
            if news_times[news_idx] <= current_time:
                delta_minutes = (
                    (np.datetime64(current_time) - news_times[news_idx])
                    / np.timedelta64(1, "m")
                )

                sentiment = float(news_sentiments[news_idx])
                effective = sentiment * np.exp(-lambda_decay * delta_minutes)

                self.df.at[ts, "last_news_sentiment"] = sentiment
                self.df.at[ts, "news_age_minutes"] = float(delta_minutes)
                self.df.at[ts, "effective_sentiment_t"] = effective
                self.df.at[ts, "news_id"] = news_ids[news_idx]
                self.df.at[ts, "news_headline"] = news_headlines[news_idx]

            if (i + 1) % 100_000 == 0:
                print(
                    f"  [Progress] {i+1:,} / {total_rows:,} Bars "
                    f"({(i+1)/total_rows*100:.1f}%) verarbeitet ..."
                )

        print("[INFO] News-Alignment abgeschlossen.")

    # ------------------------------------------------------------------
    # 9. Index-Features (z.B. QQQ) + Relative Returns
    # ------------------------------------------------------------------
    def _add_index_features(self, past_return_windows: List[int]) -> None:
        """
        Fügt Index-Features (z.B. QQQ) hinzu und berechnet
        relative Log-Returns Aktie vs. Index.

        Erwartet self.index_df != None.
        """
        if self.index_df is None:
            print("[INFO] Kein index_df übergeben – keine Index-Features berechnet.")
            return

        idx = self.index_df.copy()
        # 1m Log-Return des Index
        idx["index_log_ret_1m"] = np.log(idx[self.index_price_col]).diff()

        # Aggregierte Log-Returns über dieselben Fenster wie bei der Aktie
        for w in past_return_windows:
            idx[f"index_log_ret_{w}m"] = (
                idx["index_log_ret_1m"].rolling(window=w).sum()
            )

        # Realisierte Volatilität des Index
        for w in self.realized_vol_windows:
            idx[f"index_rv_{w}m"] = (
                idx["index_log_ret_1m"]
                .rolling(window=w)
                .apply(lambda x: np.sqrt((x ** 2).sum()), raw=False)
            )

        # Nur die neuen Index-Spalten behalten
        index_cols = [c for c in idx.columns if c.startswith("index_")]

        # Join auf die Aktien-Daten (per Timestamp-Index)
        self.df = self.df.join(idx[index_cols], how="left")

        # Relative Performance: Aktie – Index, in Log-Returns
        for w in past_return_windows:
            stock_col = f"log_ret_{w}m"
            index_col = f"index_log_ret_{w}m"
            rel_col = f"rel_log_ret_{w}m"
            if stock_col in self.df.columns and index_col in self.df.columns:
                self.df[rel_col] = self.df[stock_col] - self.df[index_col]

    # ------------------------------------------------------------------
    # 10. Master-Funktion: alle Features berechnen
    # ------------------------------------------------------------------
    def build_features_before_split(
        self,
        save: bool = False,
        save_suffix: str = "",
    ) -> pd.DataFrame:
        """
        Führt alle Feature-Schritte aus und gibt den erweiterten DataFrame zurück.

        Wenn save=True:
          - speichert nach data/processed/{symbol}_features{save_suffix}.parquet/csv
        """
        # Reihenfolge bewusst: erst Returns, dann abgeleitete Größen
        past_windows = [15, 30, 60, 120]

        used_windows = self._add_log_returns(past_return_windows=past_windows)
        self._add_emas()
        self._add_ema_differences()
        self._add_zscore()
        self._add_realized_volatility()
        self._add_high_low_span()
        self._add_avg_volume_per_trade()
        self._align_news_with_price(lambda_decay=0.0075)
        # NEU: Index-Features (falls index_df vorhanden)
        self._add_index_features(used_windows)

        features_df = self.df.copy()

        if save:
            processed_dir = self.data_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            base_name = f"{self.symbol}_features{save_suffix}"
            parquet_path = processed_dir / f"{base_name}.parquet"
            csv_path = processed_dir / f"{base_name}.csv"

            features_df.to_parquet(parquet_path, index=True)

            print(f"[OK] Features gespeichert unter:\n  {parquet_path}\n  {csv_path}")

        return features_df
