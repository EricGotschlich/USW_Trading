#!/usr/bin/env python3
"""
USW_Trading - Paper Trading (Alpaca) using your trained model
============================================================

- Uses Alpaca PAPER account (no real money)
- Builds SAME features as training (FeatureBuilder)
- FFNN: uses sym_* one-hot + scaler_X.joblib
- LSTM: uses ONLY FEATURE_COLS + scaler_X_lstm.joblib (recommended with your current lstm_modeling.py)
- Optional cooldown + optional short (if account supports it)

Env vars:
  ALPACA_API_KEY
  ALPACA_API_SECRET
Optional:
  ALPACA_BASE_URL (default: https://paper-api.alpaca.markets)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from torch import nn

# Project import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.stock_feature_builder import FeatureBuilder, FeatureBuilderConfig  # type: ignore


# -----------------------------
# Paths
# -----------------------------
ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)



# -----------------------------
# News cache (avoid re-reading huge parquet every loop)
# -----------------------------
NEWS_CACHE_DF = None
NEWS_CACHE_TS_COL = None
NEWS_CACHE_LOADED_AT = 0.0
NEWS_CACHE_TTL_SEC = 300  # reload every 5 minutes (optional)

def _load_news_cache(project_root: Path):
    """
    Loads the parquet once and detects the timestamp column.
    Returns (df, ts_col).
    """
    global NEWS_CACHE_DF, NEWS_CACHE_TS_COL, NEWS_CACHE_LOADED_AT

    path = project_root / "data" / "processed" / "nasdaq_news_with_sentiment.parquet"
    if not path.exists():
        # If you keep it elsewhere, adjust here
        raise FileNotFoundError(f"News parquet not found: {path}")

    df = pd.read_parquet(path)

    ts_col = None
    for c in ["created_at", "providerPublishTime", "timestamp", "updated_at"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"Could not find a timestamp column in news parquet. Columns={list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col])
    df[ts_col] = df[ts_col].dt.tz_convert("UTC").dt.tz_localize(None)

    # normalize symbol
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()

    NEWS_CACHE_DF = df
    NEWS_CACHE_TS_COL = ts_col
    NEWS_CACHE_LOADED_AT = time.time()
    return df, ts_col


def get_news_cache(project_root: Path):
    global NEWS_CACHE_DF, NEWS_CACHE_TS_COL, NEWS_CACHE_LOADED_AT
    if NEWS_CACHE_DF is None or (time.time() - NEWS_CACHE_LOADED_AT) > NEWS_CACHE_TTL_SEC:
        return _load_news_cache(project_root)
    return NEWS_CACHE_DF, NEWS_CACHE_TS_COL


# -----------------------------
# Paper-trading-only FeatureBuilder override:
# only use last X minutes of news
# -----------------------------
class LiveFeatureBuilder(FeatureBuilder):
    def __init__(self, *args, news_lookback_min: int = 60, **kwargs):
        super().__init__(*args, **kwargs)
        self.news_lookback_min = int(news_lookback_min)

    def _align_news_with_price(self):
        """
        Override training builder behavior:
        - use ONLY last `news_lookback_min` minutes of news for the *latest* bar
        - do NOT iterate over full history
        """
        # default columns
        self.df["last_news_sentiment"] = 0.0
        self.df["news_age_minutes"] = np.nan
        self.df["effective_sentiment_t"] = 0.0
        self.df["news_id"] = np.nan

        # load cached news
        try:
            news, ts_col = get_news_cache(Path(self.project_root))
        except Exception as e:
            print(f"[WARN] Could not load news cache: {e} -> effective_sentiment_t=0")
            return

        if "symbol" not in news.columns or "sentiment_score" not in news.columns:
            print("[WARN] News file missing required columns (symbol/sentiment_score). -> effective_sentiment_t=0")
            return

        sym = str(self.symbol).upper()
        news_sym = news[news["symbol"] == sym].copy()
        if news_sym.empty:
            return

        # only last X minutes relative to latest bar time
        if len(self.df.index) == 0:
            return
        # --- robust: use last index key for assignment, but compare in UTC tz-aware ---
        if len(self.df.index) == 0:
            return

        idx_key = self.df.index.max()  # keep exact index type (naive/aware)
        current_time = pd.Timestamp(idx_key)

        # make current_time tz-aware UTC for comparisons
        if current_time.tzinfo is None:
            current_time_utc = current_time.tz_localize("UTC")
        else:
            current_time_utc = current_time.tz_convert("UTC")

        cutoff = current_time_utc - pd.Timedelta(minutes=self.news_lookback_min)

        # ensure news timestamps are tz-aware UTC
        news_sym = news_sym.copy()
        news_sym[ts_col] = pd.to_datetime(news_sym[ts_col], errors="coerce", utc=True)
        news_sym = news_sym.dropna(subset=[ts_col])

        # filter to lookback window
        news_sym = news_sym[(news_sym[ts_col] >= cutoff) & (news_sym[ts_col] <= current_time_utc)]
        if news_sym.empty:
            return

        news_sym = news_sym.sort_values(ts_col)
        last = news_sym.iloc[-1]

        L = float(getattr(self.config, "news_decay_lambda", 0.1386))
        news_time = pd.Timestamp(last[ts_col])  # tz-aware UTC
        age_min = (current_time_utc - news_time).total_seconds() / 60.0
        base_sent = float(last["sentiment_score"])
        eff_sent = base_sent * float(np.exp(-L * age_min))

        # assign ONLY latest row using the original index key
        self.df.at[idx_key, "last_news_sentiment"] = base_sent
        self.df.at[idx_key, "news_age_minutes"] = age_min
        self.df.at[idx_key, "effective_sentiment_t"] = eff_sent

        id_col = "id" if "id" in news_sym.columns else news_sym.columns[0]
        self.df.at[idx_key, "news_id"] = last[id_col]

        if news_sym.empty:
            return

        news_sym = news_sym.sort_values(ts_col)
        last = news_sym.iloc[-1]

        L = float(getattr(self.config, "news_decay_lambda", 0.1386))
        news_time = pd.Timestamp(last[ts_col])
        age_min = (current_time - news_time).total_seconds() / 60.0
        base_sent = float(last["sentiment_score"])
        eff_sent = base_sent * float(np.exp(-L * age_min))

        # assign ONLY latest row
        self.df.at[current_time, "last_news_sentiment"] = base_sent
        self.df.at[current_time, "news_age_minutes"] = age_min
        self.df.at[current_time, "effective_sentiment_t"] = eff_sent

        id_col = "id" if "id" in news_sym.columns else news_sym.columns[0]
        self.df.at[current_time, "news_id"] = last[id_col]



# -----------------------------
# Alpaca REST helpers
# -----------------------------
def alpaca_headers() -> Dict[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    sec = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA_API_KEY and ALPACA_API_SECRET (or ALPACA_SECRET_KEY) in environment.")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": sec,
        "Content-Type": "application/json",
    }



def alpaca_base_url() -> str:
    return os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def alpaca_data_base_url() -> str:
    return "https://data.alpaca.markets"


def get_positions() -> Dict[str, float]:
    url = f"{alpaca_base_url()}/v2/positions"
    r = requests.get(url, headers=alpaca_headers(), timeout=20)
    r.raise_for_status()
    pos = {}
    for p in r.json():
        # qty can be negative for shorts
        pos[p["symbol"]] = float(p["qty"])
    return pos


def close_position(symbol: str) -> None:
    url = f"{alpaca_base_url()}/v2/positions/{symbol}"
    r = requests.delete(
        url,
        params={"cancel_orders": "true"},  # wichtig bei bracket orders
        headers=alpaca_headers(),
        timeout=20,
    )
    if r.status_code not in (200, 204):
        print(f"[WARN] close_position failed for {symbol}: {r.status_code} {r.text}")



def submit_bracket_buy(symbol: str, notional: float, last_price: float, take_profit: float, stop_loss: float) -> None:
    if not np.isfinite(last_price) or last_price <= 0:
        print(f"[WARN] Invalid last_price for buy {symbol}: {last_price}")
        return

    qty = int(max(1.0, np.floor(notional / last_price)))  # <-- Integer qty
    url = f"{alpaca_base_url()}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": str(take_profit)},
        "stop_loss": {"stop_price": str(stop_loss)},
    }
    r = requests.post(url, json=payload, headers=alpaca_headers(), timeout=20)
    if r.status_code >= 300:
        print(f"[WARN] BUY order failed {symbol}: {r.status_code} {r.text}")
    else:
        print(f"[ORDER] BUY {symbol} qty={qty} TP={take_profit:.2f} SL={stop_loss:.2f}")


def submit_bracket_short(symbol: str, notional: float, last_price: float, take_profit: float, stop_loss: float) -> None:
    """
    Bracket SHORT (sell to open). Alpaca generally requires qty for sells.
    We approximate qty from notional and round down to an integer.
    """
    if not np.isfinite(last_price) or last_price <= 0:
        print(f"[WARN] Invalid last_price for short {symbol}: {last_price}")
        return

    qty = int(max(1.0, np.floor(notional / last_price)))

    url = f"{alpaca_base_url()}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": str(take_profit)},  # below entry for short
        "stop_loss": {"stop_price": str(stop_loss)},       # above entry for short
    }
    r = requests.post(url, json=payload, headers=alpaca_headers(), timeout=20)
    if r.status_code >= 300:
        print(f"[WARN] SHORT order failed {symbol}: {r.status_code} {r.text}")
    else:
        print(f"[ORDER] SHORT {symbol} qty={qty} TP={take_profit:.2f} SL={stop_loss:.2f}")

def fetch_latest_trade_price(symbol: str) -> float | None:
    url = f"{alpaca_data_base_url()}/v2/stocks/{symbol}/trades/latest"
    params = {"feed": "sip"}
    r = requests.get(url, params=params, headers=alpaca_headers(), timeout=20)
    if r.status_code == 403:
        params["feed"] = "iex"
        r = requests.get(url, params=params, headers=alpaca_headers(), timeout=20)
    r.raise_for_status()
    t = r.json().get("trade", {})
    p = t.get("p", None)
    return float(p) if p is not None else None



def fetch_bars(symbol: str, limit: int = 120) -> pd.DataFrame:
    url = f"{alpaca_data_base_url()}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Min",
        "limit": limit,
        "adjustment": "raw",
        "feed": "sip",
    }
    r = requests.get(url, params=params, headers=alpaca_headers(), timeout=20)
    if r.status_code == 403 and params["feed"] == "sip":
        params["feed"] = "iex"
        r = requests.get(url, params=params, headers=alpaca_headers(), timeout=20)

    r.raise_for_status()
    data = r.json().get("bars", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.sort_values("timestamp")

    # Make timestamp naive UTC (to match your offline pipeline style)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[cols].copy()


# -----------------------------
# Models
# -----------------------------
class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def load_ffnn_model(path: Path, device: torch.device) -> nn.Module:
    sd = torch.load(path, map_location="cpu")
    weight_keys = [k for k in sd.keys() if k.endswith(".weight") and k.startswith("net.")]
    weight_keys_sorted = sorted(weight_keys, key=lambda k: int(k.split(".")[1]))
    sizes: List[int] = []
    for k in weight_keys_sorted:
        w = sd[k]
        out_dim, in_dim = w.shape
        if not sizes:
            sizes.append(in_dim)
        sizes.append(out_dim)
    model = MLP(layer_sizes=sizes, dropout=0.2).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_lstm_model(path: Path, input_size: int, device: torch.device) -> nn.Module:
    sd = torch.load(path, map_location="cpu")

    # hidden_size from weight_ih_l0: shape (4*hidden, input)
    wih_key = "lstm.weight_ih_l0" if "lstm.weight_ih_l0" in sd else [k for k in sd.keys() if k.endswith("lstm.weight_ih_l0")][0]
    wih = sd[wih_key]
    hidden_size = wih.shape[0] // 4

    # num_layers from keys
    import re
    layer_ids = []
    for k in sd.keys():
        m = re.search(r"lstm\.weight_ih_l(\d+)", k)
        if m:
            layer_ids.append(int(m.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 1

    # output_size from fc.weight
    fcw_key = "fc.weight" if "fc.weight" in sd else [k for k in sd.keys() if k.endswith("fc.weight")][0]
    output_size = sd[fcw_key].shape[0]

    model = SimpleLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# -----------------------------
# Feature building
# -----------------------------
FEATURE_COLS = [
    "log_ret_1m", "log_ret_5m", "log_ret_10m", "log_ret_15m",
    "ema_diff_5_15",
    "rv_5m", "rv_15m",
    "volume_zscore_15m",
    "avg_volume_per_trade",
    "hl_span",
    "index_log_ret_1m", "index_log_ret_15m",
    "effective_sentiment_t",
]

def align_index_to_stock_timestamps(stock_bars: pd.DataFrame, index_bars: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure index_bars (QQQ) has values for every timestamp in stock_bars.
    We forward-fill missing minutes so FeatureBuilder's join won't produce NaNs.
    Returns a DF with a 'timestamp' column again (not index).
    """
    if stock_bars.empty or index_bars.empty:
        return index_bars

    df = stock_bars.copy()
    idx = index_bars.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    idx["timestamp"] = pd.to_datetime(idx["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    idx = idx.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")

    idx = idx.set_index("timestamp")

    # Reindex to stock timestamps, forward-fill
    idx_aligned = idx.reindex(df["timestamp"], method="ffill")

    # If still NaNs at the very beginning, backfill once (rare)
    idx_aligned = idx_aligned.fillna(method="bfill")

    idx_aligned = idx_aligned.reset_index().rename(columns={"index": "timestamp"})
    return idx_aligned


def build_latest_feature_row(symbol: str, bars: pd.DataFrame, index_bars: pd.DataFrame, news_lookback_min: int) -> Optional[pd.Series]:
    if bars.empty or index_bars.empty:
        return None

    df = bars.copy()
    # Alpaca bars liefern oft kein trade_count -> für avg_volume_per_trade nötig
    if "trade_count" not in df.columns:
        df["trade_count"] = 100.0  # einfache Approximation (konstant)
    else:
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").fillna(100.0)

    idx_aligned = align_index_to_stock_timestamps(df, index_bars)

    cfg = FeatureBuilderConfig()
    fb = LiveFeatureBuilder(
        df=df,
        symbol=symbol,
        index_df=idx_aligned,  # <<< WICHTIG: aligned index bars
        timestamp_col="timestamp",
        config=cfg,
        news_lookback_min=news_lookback_min,
    )

    feat = fb.build_features_before_split(save=False)
    last = feat.iloc[-1]
    missing_cols = [c for c in FEATURE_COLS if c not in feat.columns]
    nan_cols = [c for c in FEATURE_COLS if c in feat.columns and pd.isna(last[c])]

    if missing_cols or nan_cols:
        print(f"[{symbol}] missing_cols={missing_cols} nan_in_last_row={nan_cols}")
        return None

    if "effective_sentiment_t" not in feat.columns:
        feat["effective_sentiment_t"] = 0.0

    feat = feat.replace([np.inf, -np.inf], np.nan)

    feat_sub = feat[FEATURE_COLS].dropna()
    if feat_sub.empty:
        return None

    return feat_sub.iloc[-1]

    if feat.empty:
        return None

    missing = [c for c in FEATURE_COLS if c not in feat.columns]
    if missing:
        print(f"[WARN] Missing features for {symbol}: {missing}")
        return None

    return feat.iloc[-1][FEATURE_COLS]


def add_onehots(x: pd.Series, symbols: List[str], symbol: str) -> pd.Series:
    x = x.copy()
    for s in symbols:
        x[f"sym_{s}"] = 1.0 if s == symbol else 0.0
    return x


def load_scalers(model: str):
    """
    FFNN expects scaler_X.joblib (FEATURE_COLS + sym_*)
    LSTM (with your current lstm_modeling.py) should use scaler_X_lstm.joblib (FEATURE_COLS only)
    """
    sy = joblib.load(ML_DIR / "scaler_y.joblib")

    if model == "lstm":
        sx_lstm = ML_DIR / "scaler_X_lstm.joblib"
        if sx_lstm.exists():
            sx = joblib.load(sx_lstm)
        else:
            # fallback, but likely wrong if your LSTM was trained with FEATURE_COLS-only scaling
            print(f"[WARN] Missing {sx_lstm}. Falling back to scaler_X.joblib (may mismatch your LSTM training).")
            sx = joblib.load(ML_DIR / "scaler_X.joblib")
    else:
        sx = joblib.load(ML_DIR / "scaler_X.joblib")

    return sx, sy


def compute_signal_from_pred(pred_row: np.ndarray) -> float:
    # targets: [1m, 5m, 10m, 15m]
    if pred_row.shape[0] >= 4:
        return float(0.6 * pred_row[1] + 0.4 * pred_row[3])
    return float(pred_row[0])


def _round_price_01(x: float) -> float:
    # US equities >= $1: 1-cent ticks
    return float(np.round(x, 2))

def compute_tp_sl_long(base_est: float, tp_pct: float, sl_pct: float) -> tuple[float, float]:
    tick = 0.01
    # % basierte Ziele
    tp = base_est * (1.0 + tp_pct)
    sl = base_est * (1.0 - sl_pct)

    # harte Mindestabstände (Alpaca-Regel)
    tp = max(tp, base_est + tick)
    sl = min(sl, base_est - tick)

    # runden auf gültige Ticksize
    tp = _round_price_01(tp)
    sl = _round_price_01(sl)

    # nach dem Runden nochmal sichern
    if tp < base_est + tick:
        tp = _round_price_01(base_est + tick)
    if sl > base_est - tick:
        sl = _round_price_01(base_est - tick)

    return tp, sl



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ffnn", "lstm"], default="ffnn")
    parser.add_argument("--symbols", default="AAPL,AMZN,META,NVDA,TSLA")

    # Must match LSTM training when model=lstm
    parser.add_argument("--seq-len", type=int, default=30)

    parser.add_argument("--bar-lookback", type=int, default=120)
    parser.add_argument("--entry-threshold", type=float, default=0.0008)
    parser.add_argument("--exit-threshold", type=float, default=0.0003)

    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown minutes after closing a position (per symbol).")
    parser.add_argument("--max-hold", type=int, default=15,
                        help="Maximale Haltedauer in Minuten (Position wird dann geschlossen).")

    parser.add_argument("--allow-short", action="store_true", help="Try to short when signal < -entry_threshold.")

    parser.add_argument("--notional", type=float, default=1000, help="USD per trade (approx).")
    parser.add_argument("--tp-pct", type=float, default=0.007)
    parser.add_argument("--sl-pct", type=float, default=0.003)

    parser.add_argument("--sleep-sec", type=int, default=60)
    parser.add_argument("--news-lookback-min", type=int, default=60,
                        help="Use only last X minutes of news for effective_sentiment_t.")

    parser.add_argument("--log-file", default="paper_trading_log.csv")
    parser.add_argument("--dry-run", action="store_true", help="Do not send orders, only print decisions.")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    scaler_X, scaler_y = load_scalers(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "ffnn":
        model_path = MODELS_DIR / "mlp" / "best_mlp_model.pt"
        model = load_ffnn_model(model_path, device=device)
    else:
        model_path = MODELS_DIR / "lstm" / "best_lstm_model.pth"
        model = None  # lazy load after we know input_size

    # For LSTM: rolling buffer of SCALED feature vectors (per symbol)
    buffers: Dict[str, List[np.ndarray]] = {s: [] for s in symbols}

    # Cooldown tracking (epoch seconds)
    cooldown_until: Dict[str, float] = {s: 0.0 for s in symbols}
    open_since: Dict[str, Optional[pd.Timestamp]] = {s: None for s in symbols}
    prev_qty: Dict[str, float] = {s: 0.0 for s in symbols}

    log_path = LOG_DIR / args.log_file
    if not log_path.exists():
        pd.DataFrame(columns=[
            "timestamp", "symbol", "signal",
            "pred_1m", "pred_5m", "pred_10m", "pred_15m",
            "action", "position_qty", "price"
        ]).to_csv(log_path, index=False)

    print(f"[INFO] Paper trading on: {alpaca_base_url()}")
    print(f"[INFO] Model={args.model} | symbols={symbols} | dry_run={args.dry_run}")
    print(f"[INFO] seq_len={args.seq_len} | cooldown={args.cooldown}m | allow_short={args.allow_short}")
    print(f"[INFO] Logging to: {log_path}")

    while True:
        now_ts = pd.Timestamp.utcnow()

        try:
            positions = get_positions()
        except Exception as e:
            print(f"[WARN] Could not fetch positions: {e}")
            positions = {}

        idx_bars = fetch_bars("QQQ", limit=args.bar_lookback)

        for sym in symbols:
            bars = fetch_bars(sym, limit=args.bar_lookback)
            last_price = float(bars["close"].iloc[-1]) if not bars.empty else float("nan")
            qty = float(positions.get(sym, 0.0))  # can be negative for shorts

            # -------------------------
            # (A) Entry/Exit erkennen -> open_since setzen / löschen
            # -------------------------
            if prev_qty[sym] == 0.0 and qty != 0.0:
                # Position wurde neu eröffnet (z.B. Fill von BUY/SHORT)
                open_since[sym] = now_ts

            elif prev_qty[sym] != 0.0 and qty == 0.0:
                # Position ist weg (z.B. TP/SL oder manueller Close)
                open_since[sym] = None

                # Optional (empfohlen): cooldown auch bei automatischem Exit (TP/SL) starten
                if args.cooldown > 0:
                    cooldown_until[sym] = max(cooldown_until[sym], time.time() + args.cooldown * 60.0)

            prev_qty[sym] = qty

            # -------------------------
            # (B) MAX HOLD erzwingen (auch wenn Features fehlen)
            # -------------------------
            if qty != 0.0 and open_since[sym] is not None and args.max_hold > 0:
                age_min = (now_ts - open_since[sym]).total_seconds() / 60.0
                if age_min >= args.max_hold:
                    action = "CLOSE_MAX_HOLD"
                    if not args.dry_run:
                        close_position(sym)
                    if args.cooldown > 0:
                        cooldown_until[sym] = time.time() + args.cooldown * 60.0

                    # log (pred/signal unknown here)
                    row = {
                        "timestamp": now_ts.isoformat(),
                        "symbol": sym,
                        "signal": np.nan,
                        "pred_1m": np.nan, "pred_5m": np.nan, "pred_10m": np.nan, "pred_15m": np.nan,
                        "action": action,
                        "position_qty": float(qty),
                        "price": float(last_price),
                    }
                    pd.DataFrame([row]).to_csv(log_path, mode="a", header=False, index=False)

                    print(f"[{sym}] max-hold reached ({age_min:.1f}m) -> CLOSE")
                    continue

            # -------------------------
            # (C) Cooldown gate (nur für neue Entries)
            # -------------------------
            if qty == 0 and time.time() < cooldown_until[sym]:
                action = "COOLDOWN"
                row = {
                    "timestamp": now_ts.isoformat(),
                    "symbol": sym,
                    "signal": np.nan,
                    "pred_1m": np.nan, "pred_5m": np.nan, "pred_10m": np.nan, "pred_15m": np.nan,
                    "action": action,
                    "position_qty": float(qty),
                    "price": float(last_price),
                }
                pd.DataFrame([row]).to_csv(log_path, mode="a", header=False, index=False)
                print(f"[{sym}] cooldown active -> skip")
                continue

            # -------------------------
            # (D) Erst jetzt Features bauen
            # -------------------------
            feat_row = build_latest_feature_row(sym, bars, idx_bars, args.news_lookback_min)
            if feat_row is None:
                action = "NO_FEATURES"
                row = {
                    "timestamp": now_ts.isoformat(),
                    "symbol": sym,
                    "signal": np.nan,
                    "pred_1m": np.nan, "pred_5m": np.nan, "pred_10m": np.nan, "pred_15m": np.nan,
                    "action": action,
                    "position_qty": float(qty),
                    "price": float(last_price),
                }
                pd.DataFrame([row]).to_csv(log_path, mode="a", header=False, index=False)
                print(f"[{sym}] missing features -> skip decisions")
                continue

            # Build input vector
            if args.model == "ffnn":
                x = add_onehots(feat_row, symbols, sym)

                # order must match scaler_X training order
                if hasattr(scaler_X, "feature_names_in_"):
                    cols = list(scaler_X.feature_names_in_)
                else:
                    cols = FEATURE_COLS + [f"sym_{s}" for s in symbols]

                x_vec = x.reindex(cols).astype(float).to_numpy().reshape(1, -1)
                x_df = pd.DataFrame(x_vec, columns=cols)
                x_scaled = scaler_X.transform(x_df).astype(np.float32)

                with torch.no_grad():
                    pred_scaled = model(torch.from_numpy(x_scaled).to(device)).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled)[0]

            else:
                # LSTM: NO onehots (matches your current lstm_modeling.py setup)
                x_vec = feat_row.reindex(FEATURE_COLS).astype(float).to_numpy().reshape(1, -1)
                x_scaled = scaler_X.transform(x_vec).astype(np.float32)

                buf = buffers[sym]
                buf.append(x_scaled[0])
                if len(buf) > args.seq_len:
                    buf.pop(0)
                if len(buf) < args.seq_len:
                    print(f"[{sym}] building sequence buffer {len(buf)}/{args.seq_len}")
                    continue

                x_seq = np.stack(buf, axis=0).reshape(1, args.seq_len, -1).astype(np.float32)

                if model is None:
                    model = load_lstm_model(model_path, input_size=x_seq.shape[2], device=device)

                with torch.no_grad():
                    pred_scaled = model(torch.from_numpy(x_seq).to(device)).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled)[0]

            signal = compute_signal_from_pred(pred)

            # Trading decision
            action = "HOLD"
            if qty == 0:
                # Enter long
                if signal > args.entry_threshold:
                    action = "BUY"
                    if not args.dry_run:
                        # konservative Base-Schätzung (weil Fill oft etwas höher sein kann)
                        trade_px = fetch_latest_trade_price(sym)
                        base_est = trade_px if trade_px is not None else last_price
                        tp, sl = compute_tp_sl_long(base_est, args.tp_pct, args.sl_pct)

                        print(f"[DEBUG] {sym} last={last_price:.2f} base_est={base_est:.2f} TP={tp:.2f} SL={sl:.2f}")

                        submit_bracket_buy(sym, notional=args.notional, last_price=base_est, take_profit=tp, stop_loss=sl)



                # Enter short
                elif args.allow_short and signal < -args.entry_threshold:
                    action = "SHORT"
                    if not args.dry_run:
                        tp = last_price * (1.0 - args.tp_pct)  # profit if price goes down
                        sl = last_price * (1.0 + args.sl_pct)  # loss if price goes up
                        submit_bracket_short(sym, notional=args.notional, last_price=last_price, take_profit=tp, stop_loss=sl)

            else:
                # Exit logic (simple signal-based)
                if qty > 0 and signal < args.exit_threshold:
                    action = "CLOSE_LONG"
                    if not args.dry_run:
                        close_position(sym)
                    if args.cooldown > 0:
                        cooldown_until[sym] = time.time() + args.cooldown * 60.0

                if qty < 0 and signal > -args.exit_threshold:
                    action = "CLOSE_SHORT"
                    if not args.dry_run:
                        close_position(sym)
                    if args.cooldown > 0:
                        cooldown_until[sym] = time.time() + args.cooldown * 60.0

            # Log
            row = {
                "timestamp": now_ts.isoformat(),
                "symbol": sym,
                "signal": float(signal),
                "pred_1m": float(pred[0]) if pred.shape[0] > 0 else np.nan,
                "pred_5m": float(pred[1]) if pred.shape[0] > 1 else np.nan,
                "pred_10m": float(pred[2]) if pred.shape[0] > 2 else np.nan,
                "pred_15m": float(pred[3]) if pred.shape[0] > 3 else np.nan,
                "action": action,
                "position_qty": float(qty),
                "price": float(last_price),
            }
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=False, index=False)

            print(f"[{sym}] signal={signal:+.6f} qty={qty:+.2f} action={action}")

        time.sleep(max(5, args.sleep_sec))


if __name__ == "__main__":
    main()
