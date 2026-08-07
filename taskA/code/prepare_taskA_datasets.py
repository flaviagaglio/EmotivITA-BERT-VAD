"""
Prepares the Task A datasets from the original EmoITA files.

- Splits "Development set.csv" into train/val (80/20).
- Uses "Test set - Gold labels.csv" only as the final test set (not used in training).

Output in taskA/data:
- train_valence.csv / val_valence.csv / test_valence.csv
- train_arousal.csv / val_arousal.csv / test_arousal.csv
- train_dominance.csv / val_dominance.csv / test_dominance.csv
"""
from pathlib import Path

import pandas as pd

TASKA_DATA = Path(__file__).resolve().parents[1] / "data"
DEV_PATH = TASKA_DATA / "Development set.csv"
TEST_PATH = TASKA_DATA / "Test set - Gold labels.csv"

TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

DIMENSIONS = [("valence", "V"), ("arousal", "A"), ("dominance", "D")]


def main() -> None:
    dev = pd.read_csv(DEV_PATH)
    dev = dev.dropna(subset=["text", "V", "A", "D"])

    train_dev = dev.sample(frac=TRAIN_FRACTION, random_state=RANDOM_STATE)
    val_dev = dev.drop(train_dev.index)

    print(f"Train rows: {len(train_dev)}")
    print(f"Val rows:   {len(val_dev)}")

    for name, col in DIMENSIONS:
        train_dev[["text", col]].to_csv(TASKA_DATA / f"train_{name}.csv", index=False)
        val_dev[["text", col]].to_csv(TASKA_DATA / f"val_{name}.csv", index=False)

    test = pd.read_csv(TEST_PATH).dropna(subset=["text", "V", "A", "D"])
    for name, col in DIMENSIONS:
        test[["text", col]].to_csv(TASKA_DATA / f"test_{name}.csv", index=False)

    print(f"Train/val/test files for V, A, D created in {TASKA_DATA}")


if __name__ == "__main__":
    main()
