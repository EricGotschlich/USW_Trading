"""
LSTM-Modellierung (USW_Trading, schnelle Variante)
--------------------------------------------------
- Nutzt Train/Val/Test-Splits aus data/processed/splits/usw_*_clean.parquet
- Verwendet die in post_split_scale.py gelernten StandardScaler (X und y)
- Baut Sequenzen pro Symbol (nur Vergangenheit) mit Sequenzlänge = 20
- Trainiert ein kompaktes LSTM (1 Layer, hidden_size=128) für 4 Return-Horizonte
- Nutzt Early Stopping + ReduceLROnPlateau für schnelleres Training
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler  # für den eigenen X-Scaler
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme


# -------------------------------------------------
# Pfade & Konstanten
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../USW_Trading
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
ML_DIR = PROCESSED_DIR / "ml"

MODELS_DIR = PROJECT_ROOT / "models" / "lstm"
IMAGES_DIR = PROJECT_ROOT / "images" / "modeling"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Feature- und Target-Spalten identisch zu split_dataset.py / post_split_scale.py
FEATURE_COLS = [
    # Trend & Returns
    "log_ret_1m", "log_ret_5m", "log_ret_10m", "log_ret_15m",

    # EMA
    "ema_diff_5_15",

    # Volatilität (reduziert)
    "rv_5m",
    "rv_15m",

    # Volumen / Liquidität
    "volume_zscore_15m",
    "avg_volume_per_trade",
    "hl_span",

    # Markt
    "index_log_ret_1m",
    "index_log_ret_15m",

    # News
    "effective_sentiment_t",
]

TARGET_COLS = [
    "target_return_1m",
    "target_return_5m",
    "target_return_10m",
    "target_return_15m",
]


# -------------------------------------------------
# Dataset & Sequenzen
# -------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        """
        X_seq: (n_samples, seq_len, n_features)
        y_seq: (n_samples, n_targets)
        """
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y_seq).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_sequences(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    seq_len: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baut Sequenzen nur aus der Vergangenheit:
      Input:  X[t-seq_len+1 .. t]
      Target: y[t]

    X_arr: (N, n_features)
    y_arr: (N, n_targets)
    """
    X_seq, y_seq = [], []

    if len(X_arr) < seq_len:
        # zu wenig Daten
        return (
            np.empty((0, seq_len, X_arr.shape[1]), dtype=np.float32),
            np.empty((0, y_arr.shape[1]), dtype=np.float32),
        )

    for i in range(seq_len - 1, len(X_arr)):
        X_seq.append(X_arr[i - seq_len + 1 : i + 1])
        y_seq.append(y_arr[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def build_sequences_for_split(
    df_split: pd.DataFrame,
    scaler_X,
    scaler_y,
    seq_len: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    - gruppiert nach 'symbol'
    - skaliert Features/Targets mit globalen Scaler-Objekten
    - baut Sequenzen pro Symbol und konkateniert alles
    """
    all_X, all_y = [], []

    # sicherstellen, dass Zeitindex aufsteigend ist
    df_split = df_split.sort_index().copy()

    if "symbol" not in df_split.columns:
        raise ValueError("Spalte 'symbol' wird benötigt, um Sequenzen pro Aktie zu bauen.")

    for symbol, df_sym in df_split.groupby("symbol"):
        feat = df_sym[FEATURE_COLS].values
        targ = df_sym[TARGET_COLS].values

        # Skalierung wie in post_split_scale.py
        feat_scaled = scaler_X.transform(feat)
        targ_scaled = scaler_y.transform(targ)

        X_seq, y_seq = create_sequences(feat_scaled, targ_scaled, seq_len=seq_len)
        if X_seq.size == 0:
            continue

        all_X.append(X_seq)
        all_y.append(y_seq)

    if not all_X:
        raise RuntimeError("Keine Sequenzen erzeugt (zu wenig Daten je Symbol?).")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    return X_all, y_all


# -------------------------------------------------
# LSTM-Modell (kompakte Variante)
# -------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 4,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        last_layer_h = h_n[-self.num_directions :, :, :]  # (num_directions, batch, hidden_size)
        last_layer_h = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)
        return self.fc(last_layer_h)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    # --- Dateien prüfen ---
    train_path = SPLITS_DIR / "usw_train_clean.parquet"
    val_path = SPLITS_DIR / "usw_validation_clean.parquet"
    test_path = SPLITS_DIR / "usw_test_clean.parquet"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Split-Datei fehlt: {p}. "
                "Bitte zuerst split_dataset.py und post_split_scale.py ausführen."
            )

    # Nur y-Scaler aus Step 5 wird wiederverwendet;
    # für X lernen wir einen eigenen StandardScaler auf FEATURE_COLS
    scaler_y_path = ML_DIR / "scaler_y.joblib"
    if not scaler_y_path.exists():
        raise FileNotFoundError(
            "Scaler-Datei für y fehlt. Bitte zuerst post_split_scale.py ausführen."
        )

    print("[INFO] Lade Splits + Scaler ...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # y-Scaler aus Step 5 laden (Targets sind identisch)
    scaler_y = joblib.load(scaler_y_path)

    # Eigenen X-Scaler NUR für die 13 FEATURE_COLS fitten
    scaler_X = StandardScaler()
    scaler_X.fit(train_df[FEATURE_COLS].values)

    # Index sicher als DatetimeIndex
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
            else:
                df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)

    # Feature-Checks
    missing_features = [c for c in FEATURE_COLS if c not in train_df.columns]
    missing_targets = [c for c in TARGET_COLS if c not in train_df.columns]
    if missing_features:
        raise ValueError(f"Diese Feature-Spalten fehlen im Train-Split: {missing_features}")
    if missing_targets:
        raise ValueError(f"Diese Target-Spalten fehlen im Train-Split: {missing_targets}")

    # Eigenen X-Scaler NUR auf FEATURE_COLS fitten
    scaler_X = StandardScaler()
    scaler_X.fit(train_df[FEATURE_COLS].values)


    seq_len = 30  # kürzere Sequenz für schnelleres Training
    print(f"[INFO] Baue Sequenzen mit Länge {seq_len} ...")

    X_train_seq, y_train_seq = build_sequences_for_split(train_df, scaler_X, scaler_y, seq_len=seq_len)
    X_val_seq, y_val_seq = build_sequences_for_split(val_df, scaler_X, scaler_y, seq_len=seq_len)
    X_test_seq, y_test_seq = build_sequences_for_split(test_df, scaler_X, scaler_y, seq_len=seq_len)

    print("[INFO] Sequenz-Shapes:")
    print(f"  X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    print(f"  X_val_seq:   {X_val_seq.shape},   y_val_seq:   {y_val_seq.shape}")
    print(f"  X_test_seq:  {X_test_seq.shape},  y_test_seq:  {y_test_seq.shape}")

    train_dataset = SequenceDataset(X_train_seq, y_train_seq)
    val_dataset = SequenceDataset(X_val_seq, y_val_seq)
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)

    batch_size = 32  # guter Kompromiss: schnell, aber noch RAM-freundlich
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Modell & Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Verwende Device: {device}")

    n_features = X_train_seq.shape[2]
    n_outputs = y_train_seq.shape[1]

    model = LSTMModel(
        input_size=n_features,
        hidden_size=128,   # kleineres Modell
        num_layers=2,      # nur 1 Layer -> schneller
        output_size=n_outputs,
        bidirectional=False,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # höhere Lernrate für schnellere Konvergenz
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,  # schnellere LR-Reduktion
    )

    EPOCHS = 25      # weniger maximale Epochen
    patience = 1     # Early Stopping früher
    min_delta = 1e-5

    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    train_losses = []
    val_losses = []

    print("\n[INFO] Starte LSTM-Training (kompakte Variante) ...")
    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        sum_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            sum_train_loss += loss.item() * xb.size(0)

        epoch_train_loss = sum_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # ---- Validation ----
        model.eval()
        sum_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_loss = criterion(preds, yb)
                sum_val_loss += val_loss.item() * xb.size(0)
        epoch_val_loss = sum_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f}"
        )

        # Early Stopping
        if epoch_val_loss + min_delta < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> Keine Verbesserung, patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"[INFO] Early Stopping nach {epoch+1} Epochen "
                      f"(best Val Loss={best_val_loss:.6f}).")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[INFO] Bestes Modell mit Val Loss={best_val_loss:.6f} geladen.")

    # --- Evaluation auf Test-Set ---
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    preds_scaled = np.concatenate(all_preds, axis=0)
    y_scaled = np.concatenate(all_targets, axis=0)

    # zurück auf Originalskala (Targets)
    y_test_inv = scaler_y.inverse_transform(y_scaled)
    preds_inv = scaler_y.inverse_transform(preds_scaled)

    # ---------------- Baseline: 0-Return ----------------
    baseline_pred = np.zeros_like(y_test_inv)

    baseline_mse  = mean_squared_error(y_test_inv, baseline_pred)
    baseline_mae  = mean_absolute_error(y_test_inv, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2   = r2_score(y_test_inv, baseline_pred, multioutput="variance_weighted")

    print("\n[BASELINE] Zero-Return-Modell (Test-Set):")
    print(f"  MSE :  {baseline_mse:.6e}")
    print(f"  RMSE:  {baseline_rmse:.6e}")
    print(f"  MAE :  {baseline_mae:.6e}")
    print(f"  R^2 :  {baseline_r2:.4f}")

    # ---------------- LSTM-Metriken (wie FFNN) ----------------
    overall_mse  = mean_squared_error(y_test_inv, preds_inv)
    overall_mae  = mean_absolute_error(y_test_inv, preds_inv)
    overall_rmse = np.sqrt(overall_mse)
    overall_r2   = r2_score(y_test_inv, preds_inv, multioutput="variance_weighted")

    print("\n[METRICS] LSTM (Test-Set, echte Returns):")
    print(f"  MSE :  {overall_mse:.6e}")
    print(f"  RMSE:  {overall_rmse:.6e}")
    print(f"  MAE :  {overall_mae:.6e}")
    print(f"  R^2 :  {overall_r2:.4f}")

    # Zusätzlich wie beim FFNN: Pro Target R^2 etc.
    horizon_labels = ["1m", "5m", "10m", "15m"]
    print("\n[METRICS] Pro Target:")
    for i, h in enumerate(horizon_labels[:y_test_inv.shape[1]]):
        mse_i  = mean_squared_error(y_test_inv[:, i], preds_inv[:, i])
        mae_i  = mean_absolute_error(y_test_inv[:, i], preds_inv[:, i])
        rmse_i = np.sqrt(mse_i)
        r2_i   = r2_score(y_test_inv[:, i], preds_inv[:, i])
        print(
            f"  Target Return {h}: "
            f"MSE={mse_i:.6e} | RMSE={rmse_i:.6e} | MAE={mae_i:.6e} | R^2={r2_i:.4f}"
        )

    # Optional: Kurz noch die Test-MSE explizit wie vorher
    print(f"\n[INFO] Test-MSE (Original-Skala): {overall_mse:.6f}")

    # --- Plots: 4 Targets + Loss in einer PNG ---
    n_plot = min(200, len(y_test_inv))
    horizon_labels = ["1m", "5m", "10m", "15m"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    # 4 Targets: jeweils eigener Subplot
    n_outputs_to_plot = min(4, y_test_inv.shape[1])
    for i in range(n_outputs_to_plot):
        ax = axes_flat[i]
        ax.plot(
            y_test_inv[:n_plot, i],
            label="Actual",
            linewidth=2,
        )
        ax.plot(
            preds_inv[:n_plot, i],
            label="Predicted",
            linewidth=1.8,
            linestyle="--",
            alpha=0.8,
        )
        ax.set_title(f"LSTM: Target Return {horizon_labels[i]}", fontsize=11)
        ax.set_xlabel("Beobachtung")
        ax.set_ylabel("Return")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.4)

    # Fünfter Subplot (unten rechts): Training & Validation Loss
    loss_ax = axes_flat[-1]
    epochs_range = range(1, len(train_losses) + 1)
    loss_ax.plot(
        epochs_range,
        train_losses,
        label="Train Loss",
        linewidth=2,
    )
    loss_ax.plot(
        epochs_range,
        val_losses,
        label="Val Loss",
        linewidth=2,
    )
    loss_ax.set_title("Training & Validation Loss (LSTM)", fontsize=11)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("MSE Loss")
    loss_ax.legend(frameon=True, fontsize=9)
    loss_ax.grid(True, alpha=0.4)

    # Den ungenutzten fünften Subplot (axes_flat[-2]) entfernen
    fig.delaxes(axes_flat[-2])

    plt.tight_layout()
    out_path = IMAGES_DIR / "06_lstm_results.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] LSTM-Gesamtplot gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()
