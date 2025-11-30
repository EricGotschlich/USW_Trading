"""
FeatureBuilder für USW_Trading
------------------------------

Baut alle Pre-Split-Features für EIN Symbol (z.B. AAPL):

- 1-Minuten-Log-Returns + aggregierte Log-Returns über [15, 30, 60, 120] Minuten
- einfache Returns (Simple Returns) über [15, 30, 60, 120] Minuten
- EMAs (Close) über [15, 30, 60, 120] Minuten + EMA-Diffs (15–60, 30–120)
- realisierte Volatilität über [15, 30, 60, 120] Minuten
- Volume-Z-Score (rolling) basierend auf Volumen
- High-Low-Span
- durchschnittliches Volumen je Trade
- Index-Features (z.B. QQQ):
    - index_log_ret_1m und index_log_ret_{15,30,60,120}m
    - index_rv_{15,30,60,120}m
    - relative Log-Returns: rel_log_ret_{15,30,60,120}m = Aktie - Index
- News-Sentiment-Alignment mit exponentiellem Zerfall (pro Aktie)

Speichert pro Symbol ein Parquet/CSV in data/processed:
  <SYMBOL>_features.parquet / .csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureBuilderConfig:
    ema_windows: Optional[List[int]] = None
    realized_vol_windows: Optional[List[int]] = None
    return_windows: Optional[List[int]] = None  # für log + simple returns
    volume_zscore_window: int = 60
    news_decay_lambda: float = 0.0075  # ~90 Minuten Halbwertszeit

    def __post_init__(self):
        # EMAs passend zu deinen Prognosehorizonten
        if self.ema_windows is None:
            self.ema_windows = [15, 30, 60, 120]
        # Realisierte Volatilität auch über diese Fenster
        if self.realized_vol_windows is None:
            self.realized_vol_windows = [15, 30, 60, 120]
        # Return-Fenster (für log + simple)
        if self.return_windows is None:
            self.return_windows = [15, 30, 60, 120]


class FeatureBuilder:
    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        project_root: Optional[Path] = None,
        index_df: Optional[pd.DataFrame] = None,
        index_price_col: str = "close",
        timestamp_col: str = "timestamp",
        config: Optional[FeatureBuilderConfig] = None,
    ) -> None:
        # Symbol in Großbuchstaben (für News-Filter)
        self.symbol = symbol.upper()
        self.timestamp_col = timestamp_col
        self.index_df = index_df.copy() if index_df is not None else None
        self.index_price_col = index_price_col
        self.config = config or FeatureBuilderConfig()

        # Projektpfade relativ zum Repo
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        else:
            self.project_root = Path(project_root)

        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Preisdaten vorbereiten
        self.df = df.copy()
        if self.timestamp_col in self.df.columns:
            self.df[self.timestamp_col] = pd.to_datetime(
                self.df[self.timestamp_col], utc=True
            )
            self.df = self.df.set_index(self.timestamp_col)

        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index, utc=True)

        self.df = self.df.sort_index()

    # ------------------------------------------------------------------ #
    # Basis-Features: Log-Returns + Simple Returns
    # ------------------------------------------------------------------ #
    def _add_log_returns(self) -> None:
        """
        1m-Log-Return + aggregierte Log-Returns über konfigurierten Fenstern.
        log_ret_1m = log(close_t) - log(close_{t-1})
        log_ret_{w}m = Summe der letzten w log_ret_1m (nur Vergangenheit).
        """
        if "close" not in self.df.columns:
            raise ValueError("Spalte 'close' wird für log_ret_1m benötigt.")

        self.df["log_ret_1m"] = np.log(self.df["close"]).diff()

        windows = self.config.return_windows or []
        for w in windows:
            self.df[f"log_ret_{w}m"] = (
                self.df["log_ret_1m"].rolling(window=w).sum()
            )

    def _add_simple_returns(self, windows: List[int]) -> None:
        """
        Einfache Returns (Prozentänderung) über w Minuten:
        simple_return_{w}m = (close_t - close_{t-w}) / close_{t-w}
        """
        price_col = "close"
        if price_col not in self.df.columns:
            raise ValueError(f"Spalte '{price_col}' nicht gefunden.")
        for w in windows:
            self.df[f"simple_return_{w}m"] = self.df[price_col].pct_change(w)

    # ------------------------------------------------------------------ #
    # EMAs + EMA-Diffs
    # ------------------------------------------------------------------ #
    def _add_ema_features(self) -> None:
        """
        EMAs auf dem Close-Preis für alle konfigurierten Fenster
        und EMA-Differenzen, die zu den Prognosehorizonten passen.
        """
        price_col = "close"
        for w in self.config.ema_windows:
            self.df[f"ema_{w}"] = self.df[price_col].ewm(span=w, adjust=False).mean()

        # EMA-Differenzen, z.B. kurzfristig vs. mittelfristig / langfristig
        ema_pairs = [(15, 60), (30, 120)]
        for fast, slow in ema_pairs:
            fast_col = f"ema_{fast}"
            slow_col = f"ema_{slow}"
            diff_col = f"ema_diff_{fast}_{slow}"
            if fast_col in self.df.columns and slow_col in self.df.columns:
                self.df[diff_col] = self.df[fast_col] - self.df[slow_col]

    # ------------------------------------------------------------------ #
    # Realisierte Volatilität
    # ------------------------------------------------------------------ #
    def _add_realized_volatility(self) -> None:
        """
        Realisierte Volatilität über mehrere Fenster:
          rv_{W}m = sqrt( Sum_{i=t-W+1..t} (log_ret_1m_i)^2 ).
        """
        if "log_ret_1m" not in self.df.columns:
            self._add_log_returns()

        for w in self.config.realized_vol_windows:
            self.df[f"rv_{w}m"] = (
                self.df["log_ret_1m"]
                .rolling(window=w)
                .apply(lambda x: np.sqrt((x ** 2).sum()), raw=False)
            )

    # ------------------------------------------------------------------ #
    # Volumen-Features
    # ------------------------------------------------------------------ #
    def _add_volume_zscore(self) -> None:
        """
        Z-Score des Volumens: (volume - rolling_mean) / rolling_std
        mit reiner Vergangenheit (shift(1)).
        """
        col = "volume"
        if col not in self.df.columns:
            print(f"[WARN] Spalte '{col}' fehlt – Volume-Z-Score wird übersprungen.")
            return

        window = self.config.volume_zscore_window
        rolling_mean = self.df[col].rolling(window=window).mean().shift(1)
        rolling_std = self.df[col].rolling(window=window).std().shift(1)
        self.df[f"volume_zscore_{window}m"] = (self.df[col] - rolling_mean) / rolling_std

    def _add_trade_volume_feature(self) -> None:
        """
        Durchschnittliches Volumen je Trade: volume / trade_count
        """
        if "volume" in self.df.columns and "trade_count" in self.df.columns:
            self.df["avg_volume_per_trade"] = (
                self.df["volume"] / self.df["trade_count"].replace(0, np.nan)
            )
        else:
            print(
                "[INFO] 'volume' oder 'trade_count' nicht vorhanden – "
                "avg_volume_per_trade wird nicht berechnet."
            )

    def _add_hl_span(self) -> None:
        """
        High-Low-Span pro 1-Minuten-Bar.
        """
        if "high" in self.df.columns and "low" in self.df.columns:
            self.df["hl_span"] = self.df["high"] - self.df["low"]

    # ------------------------------------------------------------------ #
    # Index-Features (z.B. QQQ) + relative Log-Returns
    # ------------------------------------------------------------------ #
    def _add_index_features(self) -> None:
        """
        Fügt Index-Features hinzu (z.B. QQQ) und berechnet
        relative Log-Returns Aktie vs. Index.

        - index_log_ret_1m
        - index_log_ret_{15,30,60,120}m
        - index_rv_{15,30,60,120}m
        - rel_log_ret_{15,30,60,120}m = log_ret_stock - log_ret_index
        """
        if self.index_df is None:
            print("[INFO] kein Index-DataFrame übergeben – Index-Features entfallen.")
            return

        idx = self.index_df.copy()
        if "timestamp" in idx.columns:
            idx["timestamp"] = pd.to_datetime(idx["timestamp"], utc=True)
            idx = idx.set_index("timestamp")

        if not isinstance(idx.index, pd.DatetimeIndex):
            idx.index = pd.to_datetime(idx.index, utc=True)

        idx = idx.sort_index()

        if self.index_price_col not in idx.columns:
            print(
                f"[WARN] index_price_col '{self.index_price_col}' nicht in Index-Daten – "
                "Index-Features entfallen."
            )
            return

        # 1m Log-Return des Index
        idx["index_log_ret_1m"] = np.log(idx[self.index_price_col]).diff()

        # Aggregierte Log-Returns über dieselben Fenster wie bei der Aktie
        windows = self.config.return_windows or []
        for w in windows:
            idx[f"index_log_ret_{w}m"] = (
                idx["index_log_ret_1m"].rolling(window=w).sum()
            )

        # Realisierte Volatilität des Index
        for w in self.config.realized_vol_windows:
            idx[f"index_rv_{w}m"] = (
                idx["index_log_ret_1m"]
                .rolling(window=w)
                .apply(lambda x: np.sqrt((x ** 2).sum()), raw=False)
            )

        # Nur die relevanten Index-Spalten behalten
        index_cols = [
            c
            for c in idx.columns
            if c.startswith("index_log_ret_") or c.startswith("index_rv_")
        ]

        # Join auf die Aktien-Daten (per Timestamp-Index)
        self.df = self.df.join(idx[index_cols], how="left")

        # Relative Performance: Aktie – Index in Log-Returns
        for w in windows:
            stock_col = f"log_ret_{w}m"
            index_col = f"index_log_ret_{w}m"
            rel_col = f"rel_log_ret_{w}m"
            if stock_col in self.df.columns and index_col in self.df.columns:
                self.df[rel_col] = self.df[stock_col] - self.df[index_col]

    # ------------------------------------------------------------------ #
    # News-Alignment pro Aktie (mit exponential decay)
    # ------------------------------------------------------------------ #
    def _align_news_with_price(self) -> None:
        """
        Richtet News-Sentiment (FinBERT) an 1m-Preisdaten aus.
        Für jede Minute: letzte News vor t, mit exponentiellem Zerfall.
        """
        news_path = self.processed_dir / "nasdaq_news_with_sentiment.parquet"
        if not news_path.exists():
            print(f"[WARN] News-Datei {news_path} nicht gefunden – News-Features entfallen.")
            return

        news = pd.read_parquet(news_path)

        if "symbol" not in news.columns or "sentiment_score" not in news.columns:
            print("[WARN] 'symbol' oder 'sentiment_score' fehlen in News – Alignment abgebrochen.")
            return

        # Nur News für dieses Symbol (Großschreibung konsistent)
        news_sym = news[news["symbol"].str.upper() == self.symbol].copy()
        if news_sym.empty:
            print(f"[INFO] Keine News für {self.symbol} – News-Features = 0.")
            self.df["last_news_sentiment"] = 0.0
            self.df["news_age_minutes"] = np.nan
            self.df["effective_sentiment_t"] = 0.0
            self.df["news_id"] = np.nan
            return

        # Zeitspalte finden
        ts_col = None
        for c in ["created_at", "providerPublishTime", "timestamp", "updated_at"]:
            if c in news_sym.columns:
                ts_col = c
                break
        if ts_col is None:
            print("[WARN] Keine Zeitspalte in News gefunden – Alignment abgebrochen.")
            return

        news_sym[ts_col] = pd.to_datetime(news_sym[ts_col], utc=True)
        news_sym = news_sym.sort_values(ts_col).reset_index(drop=True)

        L = self.config.news_decay_lambda

        # Zielspalten initialisieren
        self.df["last_news_sentiment"] = 0.0
        self.df["news_age_minutes"] = np.nan
        self.df["effective_sentiment_t"] = 0.0
        self.df["news_id"] = np.nan

        news_idx = 0
        n_news = len(news_sym)
        total_bars = len(self.df)

        print(
            f"[INFO] Aligning {n_news:,} News-Items von {self.symbol} "
            f"mit {total_bars:,} Preis-Bars (λ={L}) ..."
        )

        # iteriere über alle Minuten (Index ist datetime)
        for i, (ts, _) in enumerate(self.df.iterrows()):
            current_time = ts

            # Pointer nach vorne bewegen, solange nächste News <= current_time
            while (
                news_idx < n_news - 1
                and news_sym.iloc[news_idx + 1][ts_col] <= current_time
            ):
                news_idx += 1

            if news_sym.iloc[news_idx][ts_col] <= current_time:
                nrow = news_sym.iloc[news_idx]
                news_time = nrow[ts_col]
                age_min = (current_time - news_time).total_seconds() / 60.0
                base_sent = float(nrow["sentiment_score"])
                eff_sent = base_sent * np.exp(-L * age_min)

                self.df.at[ts, "last_news_sentiment"] = base_sent
                self.df.at[ts, "news_age_minutes"] = age_min
                self.df.at[ts, "effective_sentiment_t"] = eff_sent

                id_col = "id" if "id" in news_sym.columns else news_sym.columns[0]
                self.df.at[ts, "news_id"] = nrow[id_col]

            if (i + 1) % 100_000 == 0:
                print(
                    f"  [Progress] {i + 1:,} / {total_bars:,} Bars "
                    f"({(i + 1) / total_bars * 100:.1f}%) verarbeitet ..."
                )

        print("[INFO] News-Alignment abgeschlossen.")

    # ------------------------------------------------------------------ #
    # Orchestrierung + Speichern
    # ------------------------------------------------------------------ #
    def build_features_before_split(
        self,
        save: bool = True,
        save_suffix: str = "",
    ) -> pd.DataFrame:
        """
        Baut alle Features und gibt den Feature-DataFrame zurück.
        Wenn save=True, werden Parquet/CSV pro Symbol unter data/processed gespeichert.
        """
        # Basis: erst Log-Returns, dann darauf aufbauende Features
        self._add_log_returns()
        self._add_simple_returns(self.config.return_windows)
        self._add_ema_features()
        self._add_realized_volatility()
        self._add_volume_zscore()
        self._add_trade_volume_feature()
        self._add_hl_span()
        self._add_index_features()
        self._align_news_with_price()

        # ---------------- Spaltenauswahl ---------------- #
        # Original-Preise
        cols_base = [
            c
            for c in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
            if c in self.df.columns
        ]

        # Log-Returns (1m + aggregiert)
        cols_log = [c for c in self.df.columns if c.startswith("log_ret_")]

        # Simple Returns
        cols_simple = [c for c in self.df.columns if c.startswith("simple_return_")]

        # EMAs und EMA-Diffs
        cols_ema = [c for c in self.df.columns if c.startswith("ema_")]

        # Realisierte Volatilität der Aktie (rv_*)
        cols_rv = [
            c
            for c in self.df.columns
            if c.startswith("rv_") and not c.startswith("index_")
        ]

        # Index-Features
        cols_index = [
            c
            for c in self.df.columns
            if c.startswith("index_log_ret_") or c.startswith("index_rv_")
        ]

        # Relative Log-Returns
        cols_rel = [c for c in self.df.columns if c.startswith("rel_log_ret_")]

        # Sonstige Features (Volumen, News)
        other_candidates = [
            f"volume_zscore_{self.config.volume_zscore_window}m",
            "avg_volume_per_trade",
            "hl_span",
            "last_news_sentiment",
            "news_age_minutes",
            "effective_sentiment_t",
            "news_id",
        ]
        cols_other = [c for c in other_candidates if c in self.df.columns]

        columns_to_keep = (
            cols_base
            + cols_log
            + cols_simple
            + cols_ema
            + cols_rv
            + cols_index
            + cols_rel
            + cols_other
        )

        features_df = self.df[columns_to_keep].copy()
        features_df.index.name = "timestamp"

        if save:
            out_parquet = (
                self.processed_dir / f"{self.symbol}_features{save_suffix}.parquet"
            )

            features_df.to_parquet(out_parquet, index=True)

            print(
                "[OK] Features gespeichert unter:\n"
                f"  {out_parquet}\n"

            )

        return features_df
