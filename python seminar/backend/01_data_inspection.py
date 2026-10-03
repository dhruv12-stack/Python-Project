"""
01_data_inspection.py
Inspect the raw Indian Job Market CSV before cleaning.

Run:
    python backend/01_data_inspection.py
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/row/indian-job-market-dataset-2025.xlsx")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Put the original CSV inside data/raw/ and rename it to indian_jobs.csv."
        )

    df = pd.read_excel(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    print("\n========== COLUMNS ==========")
    for col in df.columns:
        print(f"- {col}")

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    }).sort_values("missing_count", ascending=False)
    print(missing)

    print("\n========== DUPLICATES ==========")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n========== UNIQUE VALUES ==========")
    for col in df.columns:
        print(f"{col}: {df[col].nunique(dropna=True):,}")

    print("\n========== SAMPLE ==========")
    print(df.head().to_string())

    print("\n========== NUMERIC SUMMARY ==========")
    print(df.describe(include="all").T.to_string())

if __name__ == "__main__":
    main()
