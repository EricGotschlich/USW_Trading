# scripts/feature_target_correlation.py

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # --------------------------------------------------------
    # Pfade
    # --------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    splits_dir = project_root / "data" / "processed" / "splits"
    img_dir = project_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_csv = project_root / "data" / "feature_target_correlations.csv"
    out_png = img_dir / "feature_target_correlation_matrix.png"

    train_clean_path = splits_dir / "usw_train_clean.parquet"
    if not train_clean_path.exists():
        raise FileNotFoundError(
            f"Train-Clean-Datei nicht gefunden: {train_clean_path}\n"
            "Bitte zuerst split_dataset.py ausführen."
        )

    print(f"[INFO] Lade Train-Daten aus {train_clean_path} ...")
    df = pd.read_parquet(train_clean_path)

    # --------------------------------------------------------
    # Feature- und Target-Spalten definieren
    # (gleich wie im split_dataset.py, ggf. anpassen)
    # --------------------------------------------------------
    target_cols = [
        "target_return_1m",
        "target_return_5m",
        "target_return_10m",
        "target_return_15m",
    ]

    feature_cols = [
        # Trend & Returns Aktie
        "log_ret_1m",
        "log_ret_5m",
        "log_ret_10m",
        "log_ret_15m",

        # EMA
        "ema_diff_5_15",

        # Volatilität (reduziert)
        "rv_5m",
        "rv_15m",

        # Volumen / Liquidität
        "volume_zscore_15m",
        "avg_volume_per_trade",
        "hl_span",

        # Marktbewegung
        "index_log_ret_1m",
        "index_log_ret_15m",

        # News
        "effective_sentiment_t",
    ]

    # danach:
    target_cols = [c for c in target_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Nur Spalten nehmen, die wirklich existieren
    target_cols = [c for c in target_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not target_cols:
        raise ValueError("Keine Target-Spalten im Datensatz gefunden.")
    if not feature_cols:
        raise ValueError("Keine Feature-Spalten im Datensatz gefunden.")

    print(f"[INFO] Verwende {len(feature_cols)} Features und {len(target_cols)} Targets:")
    print("  Features:", feature_cols)
    print("  Targets :", target_cols)

    # --------------------------------------------------------
    # Korrelationsmatrix Feature vs Target
    # --------------------------------------------------------
    combined_cols = feature_cols + target_cols
    df_corr = df[combined_cols].dropna()

    corr_matrix = df_corr.corr()

    # Zeilen = Features, Spalten = Targets
    ft_corr = corr_matrix.loc[feature_cols, target_cols]

    # Nach erstem Target sortieren, damit es schön geordnet ist
    primary_target = target_cols[0]
    ft_corr = ft_corr.sort_values(by=primary_target, ascending=False)

    print("\n[INFO] Feature-Target-Korrelationen (sortiert nach "
          f"{primary_target}):")
    print(ft_corr)

    # --------------------------------------------------------
    # CSV speichern
    # --------------------------------------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ft_corr.to_csv(out_csv)
    print(f"\n[OK] Korrelationen als CSV gespeichert unter: {out_csv}")

    # --------------------------------------------------------
    # Heatmap plotten
    # --------------------------------------------------------
    plt.figure(figsize=(1.5 * len(target_cols) + 4,
                        0.5 * len(feature_cols) + 4))
    sns.heatmap(
        ft_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Pearson Correlation"},
    )
    plt.title("Feature–Target Correlation Matrix")
    plt.xlabel("Targets")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Heatmap gespeichert unter: {out_png}")


if __name__ == "__main__":
    main()
