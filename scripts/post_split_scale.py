"""
Step 5: Post-Split Preparation (USW_Trading)
--------------------------------------------
- Lädt Train/Validation/Test-Splits (usw_*_clean.parquet)
- Baut X / y (gleiche Feature-/Target-Listen wie in split_dataset.py)
- One-Hot-Encoding für 'symbol' als zusätzliche Features (sym_AMZN, ...).
- Trainiert je einen StandardScaler für X und y auf dem TRAIN-Set
- Speichert skaliert / unskaliert + Scaler-Objekte
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------
# Pfade (kompatibel zu split_dataset.py)
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../USW_Trading
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
ML_DIR = PROCESSED_DIR / "ml"
ML_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Features & Targets – identisch zu split_dataset.py
# --------------------------------------------------------------------
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


def main() -> None:
    # Erwartete Split-Dateien
    expected_files = [
        SPLITS_DIR / "usw_train_clean.parquet",
        SPLITS_DIR / "usw_validation_clean.parquet",
        SPLITS_DIR / "usw_test_clean.parquet",
    ]

    for path in expected_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Split-Datei nicht gefunden: {path}\n"
                "Bitte zuerst split_dataset.py ausführen."
            )

    print(f"[INFO] Lade Splits aus {SPLITS_DIR} ...")

    # Splits laden
    train_df = pd.read_parquet(SPLITS_DIR / "usw_train_clean.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "usw_validation_clean.parquet")
    test_df  = pd.read_parquet(SPLITS_DIR / "usw_test_clean.parquet")

    # ----------------- Symbol-One-Hot-Encoding ----------------- #
    sym_cols = []
    if "symbol" in train_df.columns:
        print("[INFO] Erzeuge One-Hot-Dummies für 'symbol' ...")

        # alle Symbole über alle Splits, damit die Dummy-Spalten konsistent sind
        all_symbols = pd.concat(
            [train_df["symbol"], val_df["symbol"], test_df["symbol"]],
            axis=0
        ).astype(str)
        unique_symbols = sorted(all_symbols.unique())
        sym_cols = [f"sym_{s}" for s in unique_symbols]

        def make_sym_dummies(df):
            d = pd.get_dummies(df["symbol"].astype(str), prefix="sym")
            # fehlende Spalten ergänzen
            for s in unique_symbols:
                col = f"sym_{s}"
                if col not in d.columns:
                    d[col] = 0.0
            # Spalten in fester Reihenfolge
            return d[[f"sym_{s}" for s in unique_symbols]].astype("float32")

        sym_train = make_sym_dummies(train_df)
        sym_val   = make_sym_dummies(val_df)
        sym_test  = make_sym_dummies(test_df)

        # ursprüngliche symbol-Spalte rauswerfen (String)
        train_df = train_df.drop(columns=["symbol"])
        val_df   = val_df.drop(columns=["symbol"])
        test_df  = test_df.drop(columns=["symbol"])

        print(f"[INFO] Symbol-Dummies: {sym_cols}")
    else:
        sym_train = sym_val = sym_test = None
        print("[INFO] Keine 'symbol'-Spalte gefunden – keine Dummies erzeugt.")

    # ----------------- Feature- und Target-Spalten bestimmen ----------------- #
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    target_cols  = [c for c in TARGET_COLS if c in train_df.columns]

    if not feature_cols:
        raise ValueError("Keine der FEATURE_COLS im Train-Split gefunden.")
    if not target_cols:
        raise ValueError("Keine der TARGET_COLS im Train-Split gefunden.")

    # symbol-Dummies zu den Features hinzufügen
    if sym_cols:
        final_feature_cols = feature_cols + sym_cols
    else:
        final_feature_cols = feature_cols

    print(f"[INFO] Numerische Features (ohne Symbol-Dummies): {feature_cols}")
    if sym_cols:
        print(f"[INFO] Symbol-Features: {sym_cols}")
    print(f"[INFO] Gesamt-Features für das Modell: {final_feature_cols}")

    # ----------------- X / y trennen ----------------- #
    X_train = train_df[feature_cols].copy()
    X_val   = val_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    if sym_cols:
        X_train = pd.concat([X_train, sym_train], axis=1)
        X_val   = pd.concat([X_val,   sym_val],   axis=1)
        X_test  = pd.concat([X_test,  sym_test],  axis=1)

    # Sicherstellen, dass Spaltenreihenfolge identisch ist
    X_train = X_train[final_feature_cols]
    X_val   = X_val[final_feature_cols]
    X_test  = X_test[final_feature_cols]

    y_train = train_df[target_cols].copy()
    y_val   = val_df[target_cols].copy()
    y_test  = test_df[target_cols].copy()

    # ----------------- Scaler für X ----------------- #
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)
    X_test_scaled  = scaler_X.transform(X_test)

    # ----------------- Scaler für y ----------------- #
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values)
    y_val_scaled   = scaler_y.transform(y_val.values)
    y_test_scaled  = scaler_y.transform(y_test.values)

    # ----------------- Unskalierte X/y speichern ----------------- #
    X_train.to_parquet(ML_DIR / "X_train.parquet")
    X_val.to_parquet(ML_DIR / "X_val.parquet")
    X_test.to_parquet(ML_DIR / "X_test.parquet")

    y_train.to_parquet(ML_DIR / "y_train.parquet")
    y_val.to_parquet(ML_DIR / "y_val.parquet")
    y_test.to_parquet(ML_DIR / "y_test.parquet")

    # ----------------- Skalierte X/y speichern ----------------- #
    pd.DataFrame(X_train_scaled, index=X_train.index, columns=final_feature_cols) \
        .to_parquet(ML_DIR / "X_train_scaled.parquet")
    pd.DataFrame(X_val_scaled, index=X_val.index, columns=final_feature_cols) \
        .to_parquet(ML_DIR / "X_val_scaled.parquet")
    pd.DataFrame(X_test_scaled, index=X_test.index, columns=final_feature_cols) \
        .to_parquet(ML_DIR / "X_test_scaled.parquet")

    pd.DataFrame(y_train_scaled, index=y_train.index, columns=target_cols) \
        .to_parquet(ML_DIR / "y_train_scaled.parquet")
    pd.DataFrame(y_val_scaled, index=y_val.index, columns=target_cols) \
        .to_parquet(ML_DIR / "y_val_scaled.parquet")
    pd.DataFrame(y_test_scaled, index=y_test.index, columns=target_cols) \
        .to_parquet(ML_DIR / "y_test_scaled.parquet")

    # ----------------- Scaler speichern ----------------- #
    joblib.dump(scaler_X, ML_DIR / "scaler_X.joblib")
    joblib.dump(scaler_y, ML_DIR / "scaler_y.joblib")

    print("\n[OK] Step 5 abgeschlossen:")
    print(f"  X-Shape Train/Val/Test: {X_train.shape}, {X_val.shape}, {X_test.shape}")
    print(f"  y-Shape Train/Val/Test: {y_train.shape}, {y_val.shape}, {y_test.shape}")
    print(f"  Dateien liegen unter: {ML_DIR}")


if __name__ == "__main__":
    main()
