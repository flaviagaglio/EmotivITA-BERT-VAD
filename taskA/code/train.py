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
from common.paths import resolve


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-target VAD regressor (Task A).")
    parser.add_argument("--config", required=True, help="Path to a config_<dimension>.yaml file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_name = cfg["tokenizer"]["name"]
    max_len = cfg.get("tokenizer", {}).get("max_len", 128)
    target_col = cfg["target_col"]

    train_dataset = EmoITADataset(resolve(cfg["dataset"]), tokenizer_name, max_len, [target_col])
    val_dataset = EmoITADataset(resolve(cfg["testset"]), tokenizer_name, max_len, [target_col])

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = BertVADRegressor(tokenizer_name, output_dim=1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    run_training(
        model, train_loader, val_loader, optimizer, device,
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"].get("patience", 3),
        model_save_path=resolve(cfg["model_save_path"]),
    )


if __name__ == "__main__":
    main()
