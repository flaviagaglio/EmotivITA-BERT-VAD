"""
Per visualizzare l'andamento della loss durante l'addestramento.
Legge i file taskB/risultati/train_losses.txt e val_losses.txt.
Eseguibile da qualsiasi directory.
"""

import matplotlib.pyplot as plt  # type: ignore

from _paths import path


def main():
    train_path = path("risultati", "train_losses.txt")
    val_path = path("risultati", "val_losses.txt")

    train_losses = []
    try:
        with open(train_path) as f:
            for line in f:
                _, loss = line.strip().split(",")
                train_losses.append(float(loss))
    except FileNotFoundError:
        print("File non trovato:", train_path, "- eseguire prima il training.")
        return

    val_losses = []
    try:
        with open(val_path) as f:
            for line in f:
                _, loss = line.strip().split(",")
                val_losses.append(float(loss))
    except FileNotFoundError:
        print("File val_losses.txt non trovato: il grafico mostrerà solo le train losses.")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="orange", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Andamento della Loss durante l'addestramento (Task B)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
