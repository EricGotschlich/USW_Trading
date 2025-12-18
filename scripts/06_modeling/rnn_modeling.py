"""
RNN Modeling (USW_Trading)
--------------------------
Einfaches RNN als zusätzlicher Vergleich zum FFNN und LSTM.

- Nutzt die in Step 5 erzeugten, skalierten X/y-Parquets:
  data/processed/ml/X_*_scaled.parquet, y_*_scaled.parquet
- Baut einfache Sliding-Window-Sequenzen (zeitliche Fenster)
- Trainiert ein BasicRNN (PyTorch) auf 4 Return-Horizonte
- Vergleicht mit Zero-Return-Baseline (MSE / RMSE / MAE / R^2)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# -----------------------------------------------------------
# Pfade & Konstanten
# -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ML_DIR = PROCESSED_DIR / "ml"

IMAGES_DIR = PROJECT_ROOT / "images" / "modeling"
MODELS_DIR = PROJECT_ROOT / "models" / "rnn"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("ggplot")


# -----------------------------------------------------------
# Basic RNN-Modell
# -----------------------------------------------------------
class BasicRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 4,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # letzten Zeitschritt nehmen
        last = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last)


# -----------------------------------------------------------
# Sequenz-Helfer
# -----------------------------------------------------------
def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """

    Wir wollen Target y[t] vorhersagen und dafür Features bis t sehen.
    Daher nehmen wir Sequenzen, die bei t enden:

      X_seq[i] = X[i+1 : i+1+seq_len]   # endet bei t = i+seq_len
      y_seq[i] = y[i+seq_len]           # Target am gleichen Zeitpunkt t
    """
    X_seq, y_seq = [], []
    n = len(X)
    for i in range(n - seq_len):
        X_seq.append(X[i + 1 : i + 1 + seq_len])  # (seq_len, n_features), endet bei i+seq_len
        y_seq.append(y[i + seq_len])              # (n_targets,)
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main() -> None:
    required_files = [
        "X_train_scaled.parquet",
        "X_val_scaled.parquet",
        "X_test_scaled.parquet",
        "y_train_scaled.parquet",
        "y_val_scaled.parquet",
        "y_test_scaled.parquet",
        "scaler_y.joblib",
    ]
    for fname in required_files:
        path = ML_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Benötigte Datei fehlt: {path}\n"
                "Bitte zuerst 03_pre_split_prep.py, split_dataset.py und post_split_scale.py ausführen."
            )

    print("[INFO] Lade Step-5-Daten aus", ML_DIR, "...")
    X_train_df = pd.read_parquet(ML_DIR / "X_train_scaled.parquet")
    X_val_df   = pd.read_parquet(ML_DIR / "X_val_scaled.parquet")
    X_test_df  = pd.read_parquet(ML_DIR / "X_test_scaled.parquet")

    y_train_df = pd.read_parquet(ML_DIR / "y_train_scaled.parquet")
    y_val_df   = pd.read_parquet(ML_DIR / "y_val_scaled.parquet")
    y_test_df  = pd.read_parquet(ML_DIR / "y_test_scaled.parquet")

    X_train = X_train_df.values.astype(np.float32)
    X_val   = X_val_df.values.astype(np.float32)
    X_test  = X_test_df.values.astype(np.float32)

    y_train = y_train_df.values.astype(np.float32)
    y_val   = y_val_df.values.astype(np.float32)
    y_test  = y_test_df.values.astype(np.float32)

    print("\n[SHAPES] (skaliert, tabular)")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # Scaler für spätere inverse_transform auf Returns
    scaler_y = joblib.load(ML_DIR / "scaler_y.joblib")

    # ---------------- Sequenzen bauen ----------------
    seq_len = 20
    print(f"\n[INFO] Erzeuge Sequenzen mit Länge {seq_len} ...")

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   seq_len)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  seq_len)

    print("\n[SHAPES] (Sequenzen)")
    print(f"  X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    print(f"  X_val_seq:   {X_val_seq.shape},   y_val_seq:   {y_val_seq.shape}")
    print(f"  X_test_seq:  {X_test_seq.shape},  y_test_seq:  {y_test_seq.shape}")

    if X_train_seq.size == 0 or X_val_seq.size == 0 or X_test_seq.size == 0:
        raise RuntimeError("Zu wenig Daten für RNN-Sequenzen – prüfe seq_len und Splits.")

    # ---------------- PyTorch Tensors ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Verwende Device: {device}")

    X_train_t = torch.from_numpy(X_train_seq).to(device)
    y_train_t = torch.from_numpy(y_train_seq).to(device)
    X_val_t   = torch.from_numpy(X_val_seq).to(device)
    y_val_t   = torch.from_numpy(y_val_seq).to(device)
    X_test_t  = torch.from_numpy(X_test_seq).to(device)
    y_test_t  = torch.from_numpy(y_test_seq).to(device)

    n_features = X_train.shape[1]
    n_outputs  = y_train.shape[1]

    model = BasicRNN(
        input_size=n_features,
        hidden_size=64,
        num_layers=1,
        output_size=n_outputs,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 100
    patience = 7
    min_delta = 1e-5
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("\n[INFO] Starte RNN-Training ...")
    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        optimizer.zero_grad()
        preds_train = model(X_train_t)
        loss_train = criterion(preds_train, y_train_t)
        loss_train.backward()
        optimizer.step()

        train_loss_value = loss_train.item()
        train_losses.append(train_loss_value)

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            preds_val = model(X_val_t)
            loss_val = criterion(preds_val, y_val_t)
        val_loss_value = loss_val.item()
        val_losses.append(val_loss_value)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss_value:.6f} | "
            f"Val Loss: {val_loss_value:.6f}"
        )

        # Early Stopping
        if val_loss_value + min_delta < best_val_loss:
            best_val_loss = val_loss_value
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> Keine Verbesserung, patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(
                    f"[INFO] Early Stopping nach {epoch+1} Epochen "
                    f"(best Val Loss={best_val_loss:.6f})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Bestes RNN mit Val Loss={best_val_loss:.6f} geladen.")

    # ---------------- Evaluation auf Test ----------------
    model.eval()
    with torch.no_grad():
        preds_test = model(X_test_t).cpu().numpy()
    y_test_scaled = y_test_t.cpu().numpy()

    # zurück auf Originalskala
    y_test_inv   = scaler_y.inverse_transform(y_test_scaled)
    preds_inv    = scaler_y.inverse_transform(preds_test)

    # Baseline: Zero-Return
    baseline_pred = np.zeros_like(y_test_inv)

    baseline_mse  = mean_squared_error(y_test_inv, baseline_pred)
    baseline_mae  = mean_absolute_error(y_test_inv, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2   = r2_score(y_test_inv, baseline_pred, multioutput="variance_weighted")

    print("\n[BASELINE] Zero-Return-Modell (RNN-Test-Set):")
    print(f"  MSE :  {baseline_mse:.6e}")
    print(f"  RMSE:  {baseline_rmse:.6e}")
    print(f"  MAE :  {baseline_mae:.6e}")
    print(f"  R^2 :  {baseline_r2:.4f}")

    # RNN-Metriken
    overall_mse  = mean_squared_error(y_test_inv, preds_inv)
    overall_mae  = mean_absolute_error(y_test_inv, preds_inv)
    overall_rmse = np.sqrt(overall_mse)
    overall_r2   = r2_score(y_test_inv, preds_inv, multioutput="variance_weighted")

    print("\n[METRICS] BasicRNN (Test-Set, echte Returns):")
    print(f"  MSE :  {overall_mse:.6e}")
    print(f"  RMSE:  {overall_rmse:.6e}")
    print(f"  MAE :  {overall_mae:.6e}")
    print(f"  R^2 :  {overall_r2:.4f}")

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

    # ---------------- Plots (Matplotlib) ----------------
    # ---------------- Kombinierter Plot: 4 Horizons + Loss ----------------
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
            alpha=0.8,
        )
        ax.set_title(f"BasicRNN: Target Return {horizon_labels[i]}")
        ax.set_xlabel("Beobachtung")
        ax.set_ylabel("Return")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, alpha=0.4)

    # ungenutzte Subplots (bis auf den letzten für Loss) leeren
    for j in range(n_outputs_to_plot, len(axes_flat) - 1):
        fig.delaxes(axes_flat[j])

    # Letzter Subplot: Training & Validation Loss
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
    loss_ax.set_title("BasicRNN: Training & Validation Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("MSE Loss")
    loss_ax.legend(frameon=True, fontsize=9)
    loss_ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = IMAGES_DIR / "06_rnn_results.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] RNN-Gesamtplot gespeichert unter: {out_path}")


    # ---------------- Modell speichern ----------------
    model_path = MODELS_DIR / "basic_rnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[INFO] RNN-Modell gespeichert unter: {model_path}")


if __name__ == "__main__":
    main()
