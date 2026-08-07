import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.metrics import regression_metrics
from common.paths import resolve


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single-target model against gold labels (Task A).")
    parser.add_argument("--config", required=True, help="Path to a config_<dimension>.yaml file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_save_path = resolve(cfg["model_save_path"])
    preds_path = model_save_path.replace(".pth", "_pred.csv")
    gold_path = resolve(cfg["testset"])
    target_col = cfg["target_col"]

    preds = pd.read_csv(preds_path)
    gold = pd.read_csv(gold_path)

    # predict.py writes the predictions file from this same gold_path CSV,
    # so rows are already aligned. This truncation only guards against an
    # accidental mismatch (e.g. a stale predictions file).
    if len(preds) != len(gold):
        print(f"[WARNING] predictions ({len(preds)} rows) and gold ({len(gold)} rows) "
              "have different lengths, truncating to the shorter one.")
    n = min(len(preds), len(gold))
    gold = gold.iloc[:n].reset_index(drop=True)
    preds = preds.iloc[:n].reset_index(drop=True)

    y_true = gold[target_col].to_numpy()
    y_pred = preds[f"{target_col}_pred"].to_numpy()
    metrics = regression_metrics(y_true, y_pred)

    out_path = model_save_path.replace(".pth", "_results.txt")
    with open(out_path, "w") as f:
        f.write(f"EmoITA model evaluation - {target_col}\n")
        f.write("===========================================\n\n")
        f.write(f"{target_col}:\n")
        f.write(f"  MAE = {metrics['mae']:.4f}\n")
        f.write(f"  Pearson r = {metrics['pearson_r']:.4f}\n\n")

    print(f"Evaluation complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
