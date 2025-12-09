"""
FFNN Modeling (USW_Trading)
---------------------------
Feed-Forward-Netz für 4 Return-Horizonte (1m, 5m, 10m, 15m).

Verbesserungen:
- Clippe y_train / y_val im skalierten Raum auf ±2σ (Outlier-Reduktion)
- Verwendet gewichtete Huber-Loss (SmoothL1) pro Horizont
- Learning-Rate-Scheduler (ReduceLROnPlateau)
- Vergleich mit Zero-Baseline + Metriken pro Target
"""

import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Reproduzierbarkeit
# -----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------
# Pfade
# -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ML_DIR = PROCESSED_DIR / "ml"

IMAGES_DIR = PROJECT_ROOT / "images"
MODELS_DIR = PROJECT_ROOT / "models" / "mlp"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Dateien prüfen
# -----------------------------------------------------------
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
            "Bitte zuerst post_split_scale.py ausführen."
        )

print(f"[INFO] Lade Daten aus {ML_DIR} ...")

# -----------------------------------------------------------
# Daten laden
# -----------------------------------------------------------
X_train_df = pd.read_parquet(ML_DIR / "X_train_scaled.parquet")
X_val_df   = pd.read_parquet(ML_DIR / "X_val_scaled.parquet")
X_test_df  = pd.read_parquet(ML_DIR / "X_test_scaled.parquet")

y_train_df = pd.read_parquet(ML_DIR / "y_train_scaled.parquet")
y_val_df   = pd.read_parquet(ML_DIR / "y_val_scaled.parquet")
y_test_df  = pd.read_parquet(ML_DIR / "y_test_scaled.parquet")

feature_cols = X_train_df.columns.tolist()
target_cols  = y_train_df.columns.tolist()

print("[INFO] Feature-Spalten:", feature_cols)
print("[INFO] Target-Spalten :", target_cols)

X_train = X_train_df.values.astype(np.float32)
X_val   = X_val_df.values.astype(np.float32)
X_test  = X_test_df.values.astype(np.float32)

y_train = y_train_df.values.astype(np.float32)
y_val   = y_val_df.values.astype(np.float32)
y_test  = y_test_df.values.astype(np.float32)

print("\n[SHAPES]")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# Scaler für spätere Rücktransformation
scaler_y = joblib.load(ML_DIR / "scaler_y.joblib")

# -----------------------------------------------------------
# Targets für Training/Validation aggressiver clippen (±2σ)
# -----------------------------------------------------------
CLIP_SIGMA = 2.0
print(f"\n[INFO] Clippe y_train/y_val im skalierten Raum auf ±{CLIP_SIGMA}σ ...")
y_train_clipped = np.clip(y_train, -CLIP_SIGMA, CLIP_SIGMA)
y_val_clipped   = np.clip(y_val,   -CLIP_SIGMA, CLIP_SIGMA)

# Test bleibt unverändert
y_test_clipped = y_test.copy()

# -----------------------------------------------------------
# Tensors & DataLoader
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Verwende Device: {device}")

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train_clipped)
X_val_tensor   = torch.from_numpy(X_val)
y_val_tensor   = torch.from_numpy(y_val_clipped)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test_clipped)

batch_size = 4096

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
val_ds   = TensorDataset(X_val_tensor,   y_val_tensor)
test_ds  = TensorDataset(X_test_tensor,  y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          pin_memory=torch.cuda.is_available())

# -----------------------------------------------------------
# MLP definieren
# -----------------------------------------------------------
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]

hidden1 = 64   # etwas kleiner, um Overfitting zu reduzieren
hidden2 = 32
dropout_p = 0.2

class MLP(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = MLP(in_dim, hidden1, hidden2, out_dim, dropout_p=dropout_p).to(device)
print(f"\n[INFO] Modell:\n{model}")

# -----------------------------------------------------------
# Gewichtete Huber-Loss
# -----------------------------------------------------------

# Gewichte pro Target: [1m, 5m, 10m, 15m]
# -> 1m am wichtigsten, dann 5m
loss_weights = torch.tensor([2.0, 1.5, 1.0, 1.0], device=device)  # shape (4,)
beta = 1.0  # Huber-Parameter

def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    SmoothL1/Huber pro Komponente, dann mit loss_weights gewichten.
    """
    diff = pred - target
    abs_diff = diff.abs()
    # Huber-Formel
    huber = torch.where(abs_diff < beta,
                        0.5 * (abs_diff ** 2) / beta,
                        abs_diff - 0.5 * beta)
    # (batch_size, 4) * (4,) -> (batch_size, 4)
    weighted = huber * loss_weights
    return weighted.mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# -----------------------------------------------------------
# Training mit Early Stopping
# -----------------------------------------------------------
max_epochs = 60
patience = 8   # etwas höher, weil LR-Scheduler

best_val_loss = float("inf")
epochs_no_improve = 0

train_loss_hist = []
val_loss_hist   = []

best_model_path = MODELS_DIR / "best_mlp_model.pt"

print("\n[INFO] Starte Training ...")
for epoch in range(1, max_epochs + 1):
    # ---------------- Training ----------------
    model.train()
    running_train_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(xb)
        loss = weighted_huber_loss(preds, yb)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * xb.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_loss_hist.append(epoch_train_loss)

    # ---------------- Validation ----------------
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            loss = weighted_huber_loss(preds, yb)
            running_val_loss += loss.item() * xb.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_loss_hist.append(epoch_val_loss)

    scheduler.step(epoch_val_loss)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {epoch_train_loss:.6f} | "
        f"Val Loss:   {epoch_val_loss:.6f}"
    )

    if epoch_val_loss < best_val_loss - 1e-6:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping nach Epoch {epoch}")
            break

print("\n[INFO] Training beendet.")
print(f"[INFO] Beste Validation-Loss (skaliert, gewichteter Huber): {best_val_loss:.6f}")
print(f"[INFO] Bestes Modell gespeichert unter: {best_model_path}")

# -----------------------------------------------------------
# Loss-Kurven plotten
# -----------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("FFNN: Train & Validation Loss")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()
loss_plot_path = IMAGES_DIR / "06_ffnn_loss_curves.png"
plt.savefig(loss_plot_path, dpi=200)
plt.close()
print(f"[INFO] Loss-Kurven gespeichert unter: {loss_plot_path}")

# -----------------------------------------------------------
# Test-Evaluation (echte Returns)
# -----------------------------------------------------------
print("\n[INFO] Lade bestes Modell und evaluiere auf Test-Set ...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_y_true_scaled = []
all_y_pred_scaled = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        preds = model(xb)

        all_y_true_scaled.append(yb.cpu().numpy())
        all_y_pred_scaled.append(preds.cpu().numpy())

all_y_true_scaled = np.vstack(all_y_true_scaled)
all_y_pred_scaled = np.vstack(all_y_pred_scaled)

# in echte Returns zurücktransformieren
all_y_true = scaler_y.inverse_transform(all_y_true_scaled)
all_y_pred = scaler_y.inverse_transform(all_y_pred_scaled)

# ---------------- Baseline: 0-Return ----------------
baseline_pred = np.zeros_like(all_y_true)

baseline_mse  = mean_squared_error(all_y_true, baseline_pred)
baseline_mae  = mean_absolute_error(all_y_true, baseline_pred)
baseline_rmse = math.sqrt(baseline_mse)
baseline_r2   = r2_score(all_y_true, baseline_pred, multioutput="variance_weighted")

print("\n[BASELINE] Zero-Return-Modell (Test-Set):")
print(f"  MSE :  {baseline_mse:.6e}")
print(f"  RMSE:  {baseline_rmse:.6e}")
print(f"  MAE :  {baseline_mae:.6e}")
print(f"  R^2 :  {baseline_r2:.4f}")

# ---------------- FFNN-Metriken ----------------
overall_mse  = mean_squared_error(all_y_true, all_y_pred)
overall_mae  = mean_absolute_error(all_y_true, all_y_pred)
overall_rmse = math.sqrt(overall_mse)
overall_r2   = r2_score(all_y_true, all_y_pred, multioutput="variance_weighted")

print("\n[METRICS] FFNN (Test-Set, echte Returns):")
print(f"  MSE :  {overall_mse:.6e}")
print(f"  RMSE:  {overall_rmse:.6e}")
print(f"  MAE :  {overall_mae:.6e}")
print(f"  R^2 :  {overall_r2:.4f}")

print("\n[METRICS] Pro Target:")
for i, name in enumerate(target_cols):
    mse_i  = mean_squared_error(all_y_true[:, i], all_y_pred[:, i])
    mae_i  = mean_absolute_error(all_y_true[:, i], all_y_pred[:, i])
    rmse_i = math.sqrt(mse_i)
    r2_i   = r2_score(all_y_true[:, i], all_y_pred[:, i])
    print(f"  {name}: MSE={mse_i:.6e} | RMSE={rmse_i:.6e} | MAE={mae_i:.6e} | R^2={r2_i:.4f}")

# ---------------- Plots: Actual vs Predicted ----------------
n_plot = min(500, all_y_true.shape[0])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle("FFNN: Actual vs Predicted Returns (Test-Set, erste 500 Samples)", fontsize=14)

for i, name in enumerate(target_cols):
    ax = axes[i]
    ax.plot(all_y_true[:n_plot, i], label="Actual", linewidth=1.5)
    ax.plot(all_y_pred[:n_plot, i], label="Predicted", linewidth=1.5, alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
actual_pred_plot_path = IMAGES_DIR / "06_ffnn_actual_vs_predicted.png"
plt.savefig(actual_pred_plot_path, dpi=200)
plt.close()
print(f"[INFO] Actual-vs-Predicted Plots gespeichert unter: {actual_pred_plot_path}")
