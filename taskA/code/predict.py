import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.dataset import EmoITADataset
from common.model import BertVADRegressor
from common.paths import resolve


def parse_args():
    parser = argparse.ArgumentParser(description="Run predictions with a trained single-target model (Task A).")
    parser.add_argument("--config", required=True, help="Path to a config_<dimension>.yaml file")
    return parser.parse_args()


def predict(model, data_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs * 5.0  # back to the original 1-5 scale

            if i == 0:
                print("Output shape:", outputs.shape)

            preds.extend(outputs.cpu().numpy())
    return preds


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_csv = resolve(cfg["testset"])
    target_col = cfg["target_col"]
    tokenizer_name = cfg["tokenizer"]["name"]
    max_len = cfg.get("tokenizer", {}).get("max_len", 128)
    model_save_path = resolve(cfg["model_save_path"])

    test_dataset = EmoITADataset(test_csv, tokenizer_name, max_len)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    print("Number of test sentences:", len(test_dataset))
    if len(test_dataset) == 0:
        print("The test set is empty or malformed.")
        return

    model = BertVADRegressor(tokenizer_name, output_dim=1)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    predictions = predict(model, test_loader, device)

    df_test = pd.read_csv(test_csv).reset_index(drop=True)
    df_preds = pd.DataFrame(predictions, columns=[f"{target_col}_pred"])
    df_out = pd.concat([df_test, df_preds], axis=1)
    out_path = model_save_path.replace(".pth", "_pred.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
