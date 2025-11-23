import pandas as pd
from pathlib import Path

# ==============================
# EINSTELLUNGEN FÜR VOLLE ANZEIGE
# ==============================

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)

# ==============================
# PFAD ZUR PARQUET-DATEI (RELATIV)
# ==============================

# Ordner: .../PythonProject/test/parquet test.py
# -> parents[1] = .../PythonProject
project_root = Path(__file__).resolve().parents[1]

path = project_root / "data" / "raw" / "News_raw" / "ADBE_news.parquet"

# ==============================
# PARQUET LADEN
# ==============================
df = pd.read_parquet(path)

print("Spalten im DataFrame:")
print(df.columns)
print("\n")

print("DataFrame-Info:")
print(df.info())
print("\n")

print(df.head(30).to_string(index=False))
