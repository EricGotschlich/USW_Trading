#!/usr/bin/env python3
"""
USW_Trading - Deployment Backtest (Trading Logic)
=================================================

What this does (fits your lab requirements):
- Loads your *trained model* (FFNN or LSTM) + scalers from data/processed/ml and models/.
- Creates out-of-sample predictions on the TEST split (NO look-ahead in the trading logic).
- Converts predictions into a simple trading algorithm (entry/exit rules).
- Simulates trades on real historical prices (loaded from your Prices_1m_adj/*.parquet).
- Outputs:
  - overall performance metrics (return, Sharpe-ish, max drawdown, win rate, etc.)
  - plots (equity curve vs buy&hold, example price+trades, trade-entry distribution)
  - CSV export of all trades

Run examples:
  python scripts/07_deployment/usw_backtest_trade.py --model ffnn
  python scripts/07_deployment/usw_backtest_trade.py --model lstm --seq-len 30

Notes:
- Your project targets/returns are *decimal returns* (0.01 = 1%), so we DO NOT divide by 100.
- Your ML-parquets do NOT contain prices; this script joins predictions with prices from Prices_1m_adj.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Project paths
# -----------------------------
def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    # suche nach einem Parent, der "conf/params.yaml" hat => das ist dein Repo-Root
    for p in [here] + list(here.parents):
        if (p / "conf" / "params.yaml").exists() and (p / "data").exists():
            return p
    # fallback
    return here.parents[1]

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ML_DIR = PROCESSED_DIR / "ml"
# --- weitere Basis-Pfade ---
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images" / "deployment"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)



# -----------------------------
# Helpers
# -----------------------------
def _find_prices_dir(project_root: Path) -> Path:
    """
    Locate folder that contains 1-minute adjusted price parquet files, e.g.:
      - <project>/data/raw/Prices_1m_adj
      - <project>/data/Prices_1m_adj
      - <DATA_PATH from conf/params.yaml>/Prices_1m_adj

    Returns the first directory that exists and contains at least one *.parquet.
    """

    candidates: list[Path] = []

    # 1) Try reading DATA_PATH from conf/params.yaml (matches 01data_acquisition.py)
    params_path = project_root / "conf" / "params.yaml"
    if params_path.exists():
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = yaml.safe_load(f) or {}
            data_path = (params.get("DATA_ACQUISITION") or {}).get("DATA_PATH")
            if data_path:
                p = Path(data_path)
                if not p.is_absolute():
                    p = (project_root / p).resolve()
                candidates.append(p / "Prices_1m_adj")
                candidates.append(p / "prices_1m_adj")
        except Exception:
            # If YAML fails, we still try default paths below
            pass

    # 2) Default/common locations inside repo
    candidates.extend([
        project_root / "data" / "raw" / "Prices_1m_adj",
        project_root / "data" / "Prices_1m_adj",
        project_root / "data" / "raw" / "prices_1m_adj",
        project_root / "data" / "raw" / "prices",
        project_root / "data" / "raw" / "Prices",
    ])

    # 3) Pick first existing candidate that contains parquet files
    for c in candidates:
        if c.exists() and c.is_dir():
            if any(c.glob("*.parquet")):
                return c

    # 4) Fallback: search under <project>/data for any directory that has parquet files
    data_root = project_root / "data"
    if data_root.exists():
        for p in data_root.rglob("*.parquet"):
            return p.parent

    raise FileNotFoundError(
        "Could not locate Prices_1m_adj directory.\n"
        "Tried:\n  - "
        + "\n  - ".join(str(x) for x in candidates)
        + "\n\nFix options:\n"
          "1) Run scripts/01data_acquisition.py so it creates DATA_PATH/Prices_1m_adj\n"
          "2) Check conf/params.yaml -> DATA_ACQUISITION -> DATA_PATH\n"
          "3) Or put your price parquets into one of the tried directories."
    )


def load_ml_split(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load X_<split>_scaled.parquet and y_<split>.parquet (unscaled)."""
    split = split.lower()
    x_path = ML_DIR / f"X_{split}_scaled.parquet"
    y_path = ML_DIR / f"y_{split}.parquet"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing: {y_path}")

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    # Ensure datetime index
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.to_datetime(X.index)
    if not isinstance(y.index, pd.DatetimeIndex):
        y.index = pd.to_datetime(y.index)
    return X.sort_index(), y.sort_index()


def load_scalers():
    # 1) Standard in deinem Projekt (post_split_scale.py)
    x1 = ML_DIR / "scaler_X.joblib"
    y1 = ML_DIR / "scaler_y.joblib"

    # 2) optionaler fallback (manche Teacher-Skripte speichern direkt unter data/)
    x2 = DATA_DIR / "scaler_X.joblib"
    y2 = DATA_DIR / "scaler_y.joblib"

    if x1.exists() and y1.exists():
        return joblib.load(x1), joblib.load(y1)

    if x2.exists() and y2.exists():
        return joblib.load(x2), joblib.load(y2)

    raise FileNotFoundError(
        f"Missing scaler(s). Tried:\n"
        f" - {x1} / {y1}\n"
        f" - {x2} / {y2}\n"
        f"\nRun post_split_scale.py to generate them (expected in data/processed/ml)."
    )

def infer_symbol_from_onehots(df_X: pd.DataFrame, symbols: List[str]) -> pd.Series:
    """Infer symbol name from one-hot columns sym_<TICKER>."""
    cols = [f"sym_{s}" for s in symbols]
    missing = [c for c in cols if c not in df_X.columns]
    if missing:
        raise KeyError(f"Missing one-hot columns in X: {missing}. "
                       f"Available example columns: {list(df_X.columns)[:20]}")
    sym_mat = df_X[cols].to_numpy()
    idx = sym_mat.argmax(axis=1)
    return pd.Series([symbols[i] for i in idx], index=df_X.index, name="symbol")


def load_prices(prices_dir: Path, symbol: str) -> pd.DataFrame:
    """Load 1m prices for a symbol from Prices_1m_adj/<symbol>.parquet."""
    path = prices_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prices parquet: {path}")
    df = pd.read_parquet(path)
    # common column names from Alpaca downloads
    # expecting at least: timestamp, open, high, low, close
    # sometimes timestamp could be "t"
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "t" in df.columns:
        ts = pd.to_datetime(df["t"], utc=True, errors="coerce")
    else:
        raise KeyError(f"{path} has no timestamp column. Columns: {list(df.columns)}")

    df = df.copy()
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # normalize column names
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"open", "o"}: rename_map[c] = "open"
        elif cl in {"high", "h"}: rename_map[c] = "high"
        elif cl in {"low", "l"}: rename_map[c] = "low"
        elif cl in {"close", "c"}: rename_map[c] = "close"
        elif cl in {"volume", "v"}: rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise KeyError(f"{path} missing required columns {required}. Has: {list(df.columns)}")

    # use timezone-aware UTC; your ML index is typically tz-naive -> make both UTC-naive for join
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df[["open", "high", "low", "close"]].copy()


def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)
    # tz-aware -> UTC -> tz-naiv
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return df



# -----------------------------
# Models (architecture inferred from state_dict)
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


def load_ffnn_model(path: Path, input_size: int, output_size: int, device: torch.device) -> nn.Module:
    sd = torch.load(path, map_location="cpu")
    # infer layer sizes from state_dict keys net.<idx>.weight
    weight_keys = [k for k in sd.keys() if k.endswith(".weight") and k.startswith("net.")]
    # weights appear for Linear layers only; order is net.0.weight, net.3.weight, ...
    weight_keys_sorted = sorted(weight_keys, key=lambda k: int(k.split(".")[1]))
    sizes = []
    for k in weight_keys_sorted:
        w = sd[k]
        out_dim, in_dim = w.shape
        if not sizes:
            sizes.append(in_dim)
        sizes.append(out_dim)
    if sizes[0] != input_size or sizes[-1] != output_size:
        # still usable if user changed one-hot set; but warn hard
        print(f"[WARN] Inferred FFNN sizes {sizes} but expected [{input_size}..{output_size}]. Using inferred.")
    model = MLP(layer_sizes=sizes, dropout=0.2).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_lstm_model(path: Path, input_size: int, output_size: int, device: torch.device) -> nn.Module:
    sd = torch.load(path, map_location="cpu")
    # infer hidden_size from weight_ih_l0: (4*hidden, input)
    wih = sd.get("lstm.weight_ih_l0", None)
    if wih is None:
        # maybe saved with prefix
        wih_key = [k for k in sd.keys() if k.endswith("lstm.weight_ih_l0")]
        if not wih_key:
            raise KeyError("Could not find lstm.weight_ih_l0 in state_dict.")
        wih = sd[wih_key[0]]
    hidden_size = wih.shape[0] // 4
    # infer num_layers from keys like lstm.weight_ih_l0, lstm.weight_ih_l1, ...
    layer_ids = []
    for k in sd.keys():
        m = re.search(r"lstm\.weight_ih_l(\d+)", k)
        if m:
            layer_ids.append(int(m.group(1)))
    if not layer_ids:
        raise KeyError("Could not infer num_layers from lstm.weight_ih_l* keys.")
    num_layers = max(layer_ids) + 1
    # infer output size from fc.weight
    fcw_key = "fc.weight" if "fc.weight" in sd else [k for k in sd.keys() if k.endswith("fc.weight")][0]
    out_dim = sd[fcw_key].shape[0]
    if out_dim != output_size:
        print(f"[WARN] LSTM output_size in model={out_dim} but expected {output_size}. Using inferred.")
        output_size = out_dim
    model = SimpleLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=0.2).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# -----------------------------
# Trading logic
# -----------------------------
def compute_signal(df_pred: pd.DataFrame) -> pd.Series:
    """
    Derive a *single scalar* signal from multi-horizon predictions.
    For your targets: [1m, 5m, 10m, 15m] (decimal returns)
    We'll combine 5m + 15m as a smoother decision.
    """
    cols = [c for c in df_pred.columns if c.startswith("pred_")]
    if not cols:
        raise ValueError("No pred_* columns found to compute signal.")
    # prefer 5m + 15m if available
    if "pred_target_return_5m" in cols and "pred_target_return_15m" in cols:
        sig = 0.6 * df_pred["pred_target_return_5m"] + 0.4 * df_pred["pred_target_return_15m"]
        return sig
    # else use mean
    return df_pred[cols].mean(axis=1)



@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str  # LONG or SHORT
    pnl_pct: float
    hold_minutes: int


def simulate_trades(
    df: pd.DataFrame,
    entry_threshold: float = 0.00015,
    exit_threshold: float = 0.0,
    take_profit: float = 0.006,
    stop_loss: float = 0.004,
    max_hold_minutes: int = 15,
    fee_bps: float = 0.0,
    cooldown_bars: int = 0,
    allow_short: bool = True,
) -> Tuple[List[Trade], pd.Series]:
    """
    Trading algorithm with cooldown + optional SHORT:
    - Entry LONG : signal >  entry_threshold
    - Entry SHORT: signal < -entry_threshold (if allow_short)
    - Exit LONG  : signal <  exit_threshold OR TP/SL OR max_hold
    - Exit SHORT : signal > -exit_threshold OR TP/SL OR max_hold
    - Cooldown: after any exit, wait cooldown_bars before next entry (per symbol)
    - Uses CLOSE price for fills (simplification)
    - fee_bps deducted on entry+exit (round-trip)
    """

    trades: List[Trade] = []
    equity_vals: List[float] = []
    eq = 1.0

    # position_side: 0=flat, +1=long, -1=short
    position_side = 0
    entry_price = 0.0
    entry_ts: Optional[pd.Timestamp] = None
    hold = 0

    cooldown_left = 0

    n = len(df)
    if n == 0:
        return [], pd.Series([], dtype=float, name="equity")

    for i, (ts, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        sig = float(row["signal"])
        is_last_bar = (i == n - 1)

        if position_side == 0:
            # cooldown handling
            if cooldown_left > 0:
                cooldown_left -= 1
            else:
                # entries
                if sig > entry_threshold:
                    position_side = 1
                    entry_price = price
                    entry_ts = ts
                    hold = 0
                elif allow_short and sig < -entry_threshold:
                    position_side = -1
                    entry_price = price
                    entry_ts = ts
                    hold = 0

        else:
            hold += 1

            # ✅ NEU: Hold in echten Minuten
            if entry_ts is not None:
                hold_minutes = int((ts - entry_ts).total_seconds() // 60)
            else:
                hold_minutes = int(hold)

            if position_side == 1:
                # LONG return
                ret = (price / entry_price) - 1.0
                sig_exit = sig < exit_threshold
                side_str = "LONG"
            else:
                # SHORT return (profit if price falls)
                ret = (entry_price / price) - 1.0
                sig_exit = sig > -exit_threshold
                side_str = "SHORT"

            hit_tp = ret >= take_profit
            hit_sl = ret <= -stop_loss
            time_exit = hold_minutes >= max_hold_minutes

            # force close at the end of the series so we don't leave positions open
            should_exit = hit_tp or hit_sl or time_exit or sig_exit or is_last_bar

            if should_exit:
                exit_price = price
                exit_ts = ts

                pnl_pct = ret
                pnl_pct -= 2.0 * (fee_bps / 10000.0)  # round-trip fee
                eq *= (1.0 + pnl_pct)

                trades.append(
                    Trade(
                        symbol=str(row["symbol"]),
                        entry_ts=entry_ts if entry_ts is not None else ts,
                        exit_ts=exit_ts,
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        side=side_str,
                        pnl_pct=float(pnl_pct),
                        hold_minutes=int(hold_minutes),
                    )
                )

                # reset position + start cooldown
                position_side = 0
                entry_price = 0.0
                entry_ts = None
                hold = 0
                cooldown_left = max(0, int(cooldown_bars))

        equity_vals.append(eq)

    equity_curve = pd.Series(equity_vals, index=df.index, name="equity")
    return trades, equity_curve


def compute_performance(trades: List[Trade], equity: pd.Series) -> Dict[str, float]:
    if equity.empty:
        return {"total_return": 0.0}
    total_return = float(equity.iloc[-1] - 1.0)
    # drawdown
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min())
    # win rate
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = float(wins / len(trades)) if trades else 0.0
    avg_trade = float(np.mean([t.pnl_pct for t in trades])) if trades else 0.0
    median_trade = float(np.median([t.pnl_pct for t in trades])) if trades else 0.0

    # "Sharpe-ish": use equity returns per bar (minute)
    rets = equity.pct_change().fillna(0.0).values
    if rets.std() > 1e-12:
        sharpe = float(np.mean(rets) / np.std(rets) * math.sqrt(252 * 390))  # approx trading minutes/year
    else:
        sharpe = 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "n_trades": float(len(trades)),
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "median_trade": median_trade,
        "sharpe_like": sharpe,
    }


# -----------------------------
# Prediction pipeline
# -----------------------------
def predict_ffnn(
    X_scaled: pd.DataFrame,
    model: nn.Module,
    scaler_y,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    out = []
    X_np = X_scaled.to_numpy(dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i + batch_size]).to(device)
            pred_scaled = model(xb).cpu().numpy()
            out.append(pred_scaled)
    pred_scaled_all = np.vstack(out)
    pred = scaler_y.inverse_transform(pred_scaled_all) / 100.0
    return pred


def build_sequences_per_symbol(
    X_scaled: pd.DataFrame,
    y_true: pd.DataFrame,
    symbols: List[str],
    seq_len: int,
) -> Dict[str, Dict[str, object]]:
    """
    Build LSTM sequences per symbol to avoid mixing symbols.
    Returns dict: symbol -> {"X_seq": np.ndarray, "y": np.ndarray, "index": DatetimeIndex}
    """
    data: Dict[str, Dict[str, object]] = {}

    # We still need sym_* to infer which rows belong to which symbol
    sym_series = infer_symbol_from_onehots(X_scaled, symbols)

    # IMPORTANT: LSTM was trained WITHOUT sym_* onehots -> drop them for the sequence input
    feature_cols = [c for c in X_scaled.columns if not c.startswith("sym_")]

    for sym in symbols:
        mask = sym_series == sym
        Xs = X_scaled.loc[mask].copy()
        ys = y_true.loc[mask].copy()

        # aligned indices
        idx = Xs.index.intersection(ys.index)
        Xs = Xs.loc[idx]
        ys = ys.loc[idx]

        if len(Xs) <= seq_len:
            continue

        # drop symbol onehots for LSTM input
        Xs_feat = Xs[feature_cols]

        X_np = Xs_feat.to_numpy(dtype=np.float32)
        y_np = ys.to_numpy(dtype=np.float32)

        X_seq = []
        y_seq = []
        idx_seq = []

        # aligned (B): X[i+1:i+1+L] -> y[i+L]
        for i in range(len(X_np) - seq_len):
            X_seq.append(X_np[i + 1:i + 1 + seq_len])
            y_seq.append(y_np[i + seq_len])
            idx_seq.append(Xs_feat.index[i + seq_len])

        data[sym] = {
            "X_seq": np.asarray(X_seq, dtype=np.float32),
            "y": np.asarray(y_seq, dtype=np.float32),
            "index": pd.DatetimeIndex(idx_seq),
        }

    return data



def predict_lstm(
    seq_dict: Dict[str, Dict[str, object]],
    model: nn.Module,
    scaler_y,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Predict for each symbol's sequences, then concat into a single DataFrame indexed by timestamp.
    """
    rows = []
    for sym, d in seq_dict.items():
        X_seq = d["X_seq"]
        idx = d["index"]
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                xb = torch.from_numpy(X_seq[i:i + batch_size]).to(device)
                pred_scaled = model(xb).cpu().numpy()
                preds.append(pred_scaled)
        pred_scaled_all = np.vstack(preds)
        pred = scaler_y.inverse_transform(pred_scaled_all) / 100.0

        # store
        dfp = pd.DataFrame(pred, index=idx)
        dfp["symbol"] = sym
        rows.append(dfp)

    if not rows:
        return pd.DataFrame()
    df_all = pd.concat(rows, axis=0).sort_index()
    return df_all


# -----------------------------
# Plotting
# -----------------------------
def plot_equity(equity: pd.Series, buy_hold: pd.Series, title: str, out_path: Path) -> None:
    plt.figure(figsize=(13, 5))
    plt.plot(equity.index, equity.values, label="Strategy Equity")
    plt.plot(buy_hold.index, buy_hold.values, label="Buy & Hold ")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_trade_distribution(trades: List[Trade], out_path: Path) -> None:
    if not trades:
        return
    ts = [t.entry_ts for t in trades]
    s = pd.Series(1, index=pd.DatetimeIndex(ts)).sort_index()
    # group by date
    daily = s.resample("D").sum().fillna(0)
    plt.figure(figsize=(13, 4))
    plt.bar(daily.index, daily.values)
    plt.title("Trade Entries per Day")
    plt.xlabel("Day")
    plt.ylabel("#Entries")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_example_trades(df_sym: pd.DataFrame, trades_sym: List[Trade], out_path: Path, n: int = 3) -> None:
    if df_sym.empty:
        return
    # pick first n trades
    trades_sym = trades_sym[:n]
    if not trades_sym:
        return

    # plot around each trade
    plt.figure(figsize=(13, 4 * len(trades_sym)))
    for i, tr in enumerate(trades_sym, start=1):
        ax = plt.subplot(len(trades_sym), 1, i)
        window = df_sym.loc[tr.entry_ts - pd.Timedelta(minutes=60): tr.exit_ts + pd.Timedelta(minutes=60)]
        ax.plot(window.index, window["close"].values, label="Close")
        ax.axvline(tr.entry_ts, linestyle="--", alpha=0.7, label="Entry")
        ax.axvline(tr.exit_ts, linestyle="--", alpha=0.7, label="Exit")
        ax.set_title(f"{tr.symbol} trade {i}: pnl={tr.pnl_pct*100:.2f}% hold={tr.hold_minutes}m")
        ax.grid(True, alpha=0.25)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ffnn", "lstm"], default="lstm")
    parser.add_argument("--symbols", default="AAPL,AMZN,META,NVDA,TSLA")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--entry-threshold", type=float, default=0.001)
    parser.add_argument("--exit-threshold", type=float, default=-0.0003)
    parser.add_argument("--tp", type=float, default=0.007)
    parser.add_argument("--sl", type=float, default=0.003)
    parser.add_argument("--max-hold", type=int, default=15)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown in bars (min) after closing a trade (per symbol).")
    parser.add_argument("--allow-short", action="store_true", help="Allow SHORT trades when signal < -entry_threshold.")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    prices_dir = _find_prices_dir(PROJECT_ROOT)
    print(f"[INFO] Prices dir: {prices_dir}")

    scaler_X, scaler_y = load_scalers()

    # load TEST split
    X_test_scaled, y_test = load_ml_split("test")
    X_test_scaled = _to_utc_naive_index(X_test_scaled)
    y_test = _to_utc_naive_index(y_test)

    # infer symbols from onehots
    sym_series = infer_symbol_from_onehots(X_test_scaled, symbols)

    # load prices for each symbol once
    prices: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            prices[sym] = load_prices(prices_dir, sym)
        except Exception as e:
            print(f"[WARN] Could not load prices for {sym}: {e}")

    # join X/y with prices (on timestamp) per symbol
    all_rows = []

    if args.model == "ffnn":
        model_path = MODELS_DIR / "mlp" / "best_mlp_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        input_size = X_test_scaled.shape[1]
        output_size = y_test.shape[1]
        model = load_ffnn_model(model_path, input_size=input_size, output_size=output_size, device=device)

        preds = predict_ffnn(X_test_scaled, model, scaler_y, device=device)
        df_pred = pd.DataFrame(preds, index=X_test_scaled.index, columns=[f"pred_{c}" for c in y_test.columns])
        df_pred["symbol"] = sym_series

        pred_cols = [c for c in df_pred.columns if c.startswith("pred_")]

        for sym in symbols:
            if sym not in prices:
                continue

            mask = df_pred["symbol"] == sym

            pred_sym = df_pred.loc[mask, pred_cols].copy()
            if pred_sym.empty:
                continue

            # optional: true targets nur für dieses Symbol (für Diagnose)
            y_sym = y_test.loc[mask].copy()

            px = prices[sym].copy()

            # tz-normalisieren + duplikate raus
            pred_sym = _to_utc_naive_index(pred_sym).sort_index()
            y_sym = _to_utc_naive_index(y_sym).sort_index()
            px = _to_utc_naive_index(px).sort_index()

            pred_sym = pred_sym[~pred_sym.index.duplicated(keep="last")]
            y_sym = y_sym[~y_sym.index.duplicated(keep="last")]
            px = px[~px.index.duplicated(keep="last")]

            # ✅ WICHTIG: Preise sind die Basis (volle 1-Minuten-Reihe)
            df_sym = px.join(pred_sym, how="left")

            # Predictions in Minutenlücken weitertragen (damit signal existiert)
            df_sym[pred_cols] = df_sym[pred_cols].ffill()

            # nur Zeitraum, wo es überhaupt Predictions gibt
            df_sym = df_sym.loc[pred_sym.index.min():pred_sym.index.max()]

            # Targets nur als Diagnose (niemals mit inner join!)
            df_sym = df_sym.join(y_sym, how="left", rsuffix="_true")

            df_sym["symbol"] = sym
            df_sym["signal"] = compute_signal(df_sym)

            # am Anfang ggf. noch NaNs -> raus
            df_sym = df_sym.dropna(subset=["signal", "close"])

            all_rows.append(df_sym)


    else:
        model_path = MODELS_DIR / "lstm" / "lstm" / "best_lstm_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        # build sequences per symbol (avoids mixing)
        seq_dict = build_sequences_per_symbol(X_test_scaled, y_test, symbols=symbols, seq_len=args.seq_len)
        if not seq_dict:
            raise RuntimeError("No sequences could be built. Check seq_len and symbol onehots.")
        input_size = next(iter(seq_dict.values()))["X_seq"].shape[2]
        output_size = y_test.shape[1]
        model = load_lstm_model(model_path, input_size=input_size, output_size=output_size, device=device)

        df_all = predict_lstm(seq_dict, model, scaler_y, device=device)
        if df_all.empty:
            raise RuntimeError("No LSTM predictions produced.")

        # attach column names
        pred_cols = [f"pred_{c}" for c in y_test.columns]
        df_all.columns = pred_cols + ["symbol"]

        # pred_cols existiert bei dir schon:
        # pred_cols = [f"pred_{c}" for c in y_test.columns]

        for sym in symbols:
            if sym not in prices:
                continue

            mask = df_all["symbol"] == sym
            pred_sym = df_all.loc[mask, pred_cols].copy()
            if pred_sym.empty:
                continue

            # true targets für dieses Symbol (über sym_series, nicht über intersection)
            y_sym = y_test.loc[sym_series == sym].copy()

            px = prices[sym].copy()

            pred_sym = _to_utc_naive_index(pred_sym).sort_index()
            y_sym = _to_utc_naive_index(y_sym).sort_index()
            px = _to_utc_naive_index(px).sort_index()

            pred_sym = pred_sym[~pred_sym.index.duplicated(keep="last")]
            y_sym = y_sym[~y_sym.index.duplicated(keep="last")]
            px = px[~px.index.duplicated(keep="last")]

            # ✅ Preise als Basis
            df_sym = px.join(pred_sym, how="left")
            df_sym[pred_cols] = df_sym[pred_cols].ffill()

            df_sym = df_sym.loc[pred_sym.index.min():pred_sym.index.max()]

            df_sym = df_sym.join(y_sym, how="left", rsuffix="_true")

            df_sym["symbol"] = sym
            df_sym["signal"] = compute_signal(df_sym)
            df_sym = df_sym.dropna(subset=["signal", "close"])

            all_rows.append(df_sym)

    if not all_rows:
        raise RuntimeError("No data rows after joining predictions with prices.")

    df_all = pd.concat(all_rows, axis=0).sort_index()
    # simulate trades per symbol independently, then aggregate equity multiplicatively by time order
    trades_all: List[Trade] = []
    # --- Portfolio Equity (equal-weight) ---
    base_index = df_all.index.unique().sort_values()
    equity_all = pd.Series(1.0, index=base_index)

    per_symbol_equities = []

    for sym in symbols:
        df_sym = df_all[df_all["symbol"] == sym].sort_index()
        df_sym = df_sym[~df_sym.index.duplicated(keep="last")]
        if df_sym.empty:
            continue

        trades, equity = simulate_trades(
            df_sym,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
            take_profit=args.tp,
            stop_loss=args.sl,
            max_hold_minutes=args.max_hold,
            fee_bps=args.fee_bps,
            cooldown_bars=args.cooldown,
            allow_short=args.allow_short,
        )

        trades_all.extend(trades)

        equity = equity[~equity.index.duplicated(keep="last")]
        equity = equity.reindex(base_index, method="ffill").fillna(1.0)
        per_symbol_equities.append(equity)

    # Equal-weight portfolio: average per-minute returns
    if per_symbol_equities:
        eq_mat = pd.concat(per_symbol_equities, axis=1)
        ret_mat = eq_mat.pct_change().fillna(0.0)
        port_ret = ret_mat.mean(axis=1)
        equity_all = (1.0 + port_ret).cumprod()
        equity_all.name = "equity"
    else:
        equity_all.name = "equity"

    equity_all.name = "equity"

    # buy&hold benchmark: average of each symbol close normalized
    bh_parts = []
    for sym in symbols:
        df_sym = df_all[df_all["symbol"] == sym].sort_index()
        df_sym = df_sym[~df_sym.index.duplicated(keep="last")]  # <-- NEU (wichtig)
        if df_sym.empty:
            continue

        close = df_sym["close"]
        close = close[~close.index.duplicated(keep="last")]  # <-- NEU (extra sicher)

        bh = (close / close.iloc[0]).reindex(equity_all.index, method="ffill").fillna(1.0)
        bh_parts.append(bh)

    buy_hold = pd.concat(bh_parts, axis=1).mean(axis=1) if bh_parts else equity_all.copy()
    buy_hold.name = "buy_hold"


    perf = compute_performance(trades_all, equity_all)
    print("\n" + "=" * 60)
    print(f"BACKTEST TRADE RESULTS | model={args.model} | symbols={symbols}")
    print("=" * 60)
    for k, v in perf.items():
        if k in {"win_rate", "total_return", "max_drawdown", "avg_trade", "median_trade"}:
            print(f"{k:>15}: {v*100:.3f}%")
        else:
            print(f"{k:>15}: {v:.4f}")
    print("=" * 60)

    # save trades
    if args.save_csv:
        out_trades = IMAGES_DIR / f"trades_{args.model}.csv"
        df_trades = pd.DataFrame([t.__dict__ for t in trades_all])
        df_trades.to_csv(out_trades, index=False)
        print(f"[SAVE] Trades CSV: {out_trades}")

    # plots
    title = f"Equity Curve ({args.model}) | Return={perf['total_return']*100:.2f}% | MaxDD={perf['max_drawdown']*100:.2f}% | Trades={int(perf['n_trades'])}"
    plot_equity(equity_all, buy_hold, title, IMAGES_DIR / f"equity_{args.model}.png")
    plot_trade_distribution(trades_all, IMAGES_DIR / f"trade_entries_{args.model}.png")



    # example trades per symbol
    for sym in symbols:
        df_sym = df_all[df_all["symbol"] == sym].sort_index()
        t_sym = [t for t in trades_all if t.symbol == sym]
        if not t_sym:
            continue
        plot_example_trades(df_sym, t_sym, IMAGES_DIR / f"examples_{args.model}_{sym}.png", n=3)


    print(f"[PLOT] Saved plots into: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
