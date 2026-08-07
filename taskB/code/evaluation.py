import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.metrics import regression_metrics

TASKB_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the joint V/A/D model against gold labels (Task B).")
    parser.add_argument("--config", default=str(TASKB_ROOT / "config.yaml"), help="Path to config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    preds_path = TASKB_ROOT / cfg.get("predictions_path", "risultati/my_predictions.csv")
    gold_path = TASKB_ROOT / cfg["dataset"]["gold"]
    out_path = TASKB_ROOT / cfg.get("results_path", "risultati/risultati.txt")

    preds = pd.read_csv(preds_path)
    gold = pd.read_csv(gold_path)
    n_preds, n_gold = len(preds), len(gold)

    if "id" in preds.columns and "id" in gold.columns:
        merged = pd.merge(preds, gold, on="id", how="inner", suffixes=("_pred", "_gold"))
    elif "text" in preds.columns and "text" in gold.columns:
        merged = pd.merge(preds, gold, on="text", how="inner", suffixes=("_pred", "_gold"))
    else:
        if len(preds) != len(gold):
            raise AssertionError(
                "Prediction/gold length mismatch and no common column to merge on. "
                f"Predictions: {len(preds)}, gold: {len(gold)}"
            )
        merged = pd.concat([preds.reset_index(drop=True), gold.reset_index(drop=True)], axis=1)

    n_merged = len(merged)
    if n_merged < n_preds or n_merged < n_gold:
        print(
            f"[WARNING] Merge kept {n_merged} aligned rows (pred: {n_preds}, gold: {n_gold}). "
            "Check that the test set and gold labels contain the same instances (same id or text)."
        )
    if n_merged == 0:
        raise ValueError("No overlapping rows between predictions and gold labels.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("EmoITA model evaluation\n")
        f.write("===============================\n\n")

        for dim in ["V", "A", "D"]:
            pred_col = f"{dim}_pred"
            if pred_col not in merged.columns:
                raise ValueError(f"Missing prediction column: {pred_col}")
            if dim in merged.columns:
                y_true = merged[dim].to_numpy()
            elif f"{dim}_gold" in merged.columns:
                y_true = merged[f"{dim}_gold"].to_numpy()
            else:
                raise ValueError(f"Missing gold column for dimension: {dim}")

            y_pred = merged[pred_col].to_numpy()
            # if predictions are still on the [0,1] scale, rescale to [1,5]
            if np.nanmax(y_pred) <= 1.0 + 1e-8:
                y_pred = y_pred * 5.0

            metrics = regression_metrics(y_true, y_pred)
            f.write(f"{dim}:\n")
            f.write(f"  MAE = {metrics['mae']:.4f}\n")
            f.write(f"  Pearson r = {metrics['pearson_r']:.4f}\n\n")

    print(f"Evaluation complete. Results saved to '{out_path}'")


if __name__ == "__main__":
    main()
