"""
Debug script to quickly check the structure of the CSV files in taskA/data.
Not used during training: only for inspecting columns, row counts and types.
"""
from pathlib import Path

import pandas as pd

TASKA_DATA = Path(__file__).resolve().parents[1] / "data"

for f in sorted(TASKA_DATA.glob("*.csv")):
    print(f"\n--- Checking file: {f}")
    try:
        df = pd.read_csv(f)
        print("Columns:", list(df.columns))
        print("Row count:", len(df))
        print(df.head(3))

        if "text" not in df.columns:
            print("Missing 'text' column")
        if len(df.columns) != 2:
            print("More than 2 columns found, expected only text + target")

        target_cols = [c for c in df.columns if c != "text"]
        if not target_cols:
            print("No target column found (besides 'text').")
            continue

        target_col = target_cols[0]
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            print(f"Column {target_col} is not numeric")
        else:
            print(f"Column {target_col} is numeric")

    except Exception as e:
        print("Error reading file:", e)
