import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.dataset import EmoITADataset
from common.engine import run_training, set_seed
from common.model import BertVADRegressor

TASKB_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a joint V/A/D regressor (Task B).")
    parser.add_argument("--config", default=str(TASKB_ROOT / "config.yaml"), help="Path to config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["params"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_name = cfg["tokenizer"]["name"]
    max_len = cfg["tokenizer"]["max_len"]

    train_dataset = EmoITADataset(str(TASKB_ROOT / cfg["dataset"]["train"]), tokenizer_name, max_len, ["V", "A", "D"])
    val_dataset = EmoITADataset(str(TASKB_ROOT / cfg["dataset"]["val"]), tokenizer_name, max_len, ["V", "A", "D"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["params"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["params"]["batch_size"], shuffle=False)

    model = BertVADRegressor(tokenizer_name, output_dim=3)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["params"]["lr"])

    results_dir = TASKB_ROOT / "risultati"
    results_dir.mkdir(exist_ok=True)

    run_training(
        model, train_loader, val_loader, optimizer, device,
        epochs=cfg["params"]["num_epochs"],
        patience=cfg["params"].get("patience", 3),
        model_save_path=str(results_dir / "model.pth"),
        loss_log_paths=(str(results_dir / "train_losses.txt"), str(results_dir / "val_losses.txt")),
    )


if __name__ == "__main__":
    main()
