from pathlib import Path

import matplotlib.pyplot as plt

TASKB_ROOT = Path(__file__).resolve().parents[1]


def _read_losses(path):
    losses = []
    with open(path) as f:
        for line in f:
            _, loss = line.strip().split(",")
            losses.append(float(loss))
    return losses


def main():
    train_path = TASKB_ROOT / "risultati" / "train_losses.txt"
    val_path = TASKB_ROOT / "risultati" / "val_losses.txt"

    try:
        train_losses = _read_losses(train_path)
    except FileNotFoundError:
        print("File not found:", train_path, "- run training first.")
        return

    try:
        val_losses = _read_losses(val_path)
    except FileNotFoundError:
        print("val_losses.txt not found: plotting train loss only.")
        val_losses = []

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="orange", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss curve (Task B)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = TASKB_ROOT / "risultati" / "loss_curve.png"
    plt.savefig(out_path, dpi=200)
    print("Loss curve saved to", out_path)


if __name__ == "__main__":
    main()
