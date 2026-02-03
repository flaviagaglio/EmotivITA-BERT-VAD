import pandas as pd

"""
Prepara i dataset per il Task A a partire dai file originali EmoITA.

- Parte da "Development set.csv" e fa uno split train/val (es. 80/20).
- Usa "Test set - Gold labels.csv" SOLO come test finale (non entra nel training).

Output in taskA/data:
- train_valence.csv / val_valence.csv / test_valence.csv
- train_arousal.csv / val_arousal.csv / test_arousal.csv
- train_dominance.csv / val_dominance.csv / test_dominance.csv
"""

DEV_PATH = "taskA/data/Development set.csv"
TEST_PATH = "taskA/data/Test set - Gold labels.csv"

TRAIN_FRACTION = 0.8
RANDOM_STATE = 42


def main() -> None:
    # Carica il development set (da cui faremo train/val)
    dev = pd.read_csv(DEV_PATH)

    # Teniamo solo righe con testo e tutte le colonne V, A, D non nulle
    dev = dev.dropna(subset=["text", "V", "A", "D"])

    # Split train/val sul development
    train_dev = dev.sample(frac=TRAIN_FRACTION, random_state=RANDOM_STATE)
    val_dev = dev.drop(train_dev.index)

    print(f"Righe train dev: {len(train_dev)}")
    print(f"Righe val dev:   {len(val_dev)}")

    # Salva train/val per ogni dimensione (Task A)
    for name, col in [("valence", "V"), ("arousal", "A"), ("dominance", "D")]:
        train_dev[["text", col]].to_csv(f"taskA/data/train_{name}.csv", index=False)
        val_dev[["text", col]].to_csv(f"taskA/data/val_{name}.csv", index=False)

    # Carica il test set con gold labels (non usato nel training/early stopping)
    test = pd.read_csv(TEST_PATH).dropna(subset=["text", "V", "A", "D"])

    # Salva file test per ogni dimensione (per la valutazione finale)
    for name, col in [("valence", "V"), ("arousal", "A"), ("dominance", "D")]:
        test[["text", col]].to_csv(f"taskA/data/test_{name}.csv", index=False)

    print("File train/val/test per V, A, D creati in taskA/data/")


if __name__ == "__main__":
    main()
