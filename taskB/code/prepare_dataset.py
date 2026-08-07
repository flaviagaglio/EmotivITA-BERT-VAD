from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TASKB_ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = TASKB_ROOT / "data" / "Development set.csv"
TRAIN_OUT = TASKB_ROOT / "data" / "train.csv"
VAL_OUT = TASKB_ROOT / "data" / "val.csv"

TRAIN_FRACTION = 0.8
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DEV_PATH)
    df = df.dropna(subset=["text", "V", "A", "D"])

    train_df, val_df = train_test_split(df, test_size=1 - TRAIN_FRACTION, random_state=RANDOM_STATE)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    print(f"Train size: {len(train_df)}, validation size: {len(val_df)}")


if __name__ == "__main__":
    main()
