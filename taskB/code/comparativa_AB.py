import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

TASKB_ROOT = Path(__file__).resolve().parents[1]
TASKA_RESULTS = TASKB_ROOT.parent / "taskA" / "risultati"


def parse_taskA_results():
    """Reads MAE and Pearson r for V, A, D from taskA/risultati/model_*_results.txt."""
    dim_files = [
        ("Valence", "model_valence_results.txt"),
        ("Arousal", "model_arousal_results.txt"),
        ("Dominance", "model_dominance_results.txt"),
    ]
    result = {}
    for dim, fname in dim_files:
        p = TASKA_RESULTS / fname
        if not p.exists():
            result[dim] = {"MAE": None, "Pearson r": None}
            continue
        text = p.read_text(encoding="utf-8")
        mae = None
        corr = None
        for line in text.splitlines():
            if "MAE" in line and "=" in line:
                m = re.search(r"MAE\s*=\s*([\d.]+)", line)
                if m:
                    mae = float(m.group(1))
            if "Pearson" in line and "=" in line:
                m = re.search(r"-?[\d.]+", line.split("=")[-1])
                if m:
                    corr = float(m.group(0))
        result[dim] = {"MAE": mae, "Pearson r": corr}
    return result


def parse_taskB_results():
    """Reads MAE and Pearson r for V, A, D from taskB/risultati/risultati.txt."""
    p = TASKB_ROOT / "risultati" / "risultati.txt"
    empty = {d: {"MAE": None, "Pearson r": None} for d in ("Valence", "Arousal", "Dominance")}
    if not p.exists():
        return empty

    result = dict(empty)
    text = p.read_text(encoding="utf-8")
    dim_map = {"V:": "Valence", "A:": "Arousal", "D:": "Dominance"}
    dim = None
    for line in text.splitlines():
        line = line.strip()
        if line in dim_map:
            dim = dim_map[line]
            continue
        if dim and "MAE" in line:
            m = re.search(r"MAE\s*=\s*([\d.]+)", line)
            if m:
                result[dim]["MAE"] = float(m.group(1))
        if dim and "Pearson" in line:
            m = re.search(r"-?[\d.]+", line.split("=")[-1].strip())
            if m:
                result[dim]["Pearson r"] = float(m.group(0))
    return result


def main():
    taskA = parse_taskA_results()
    taskB = parse_taskB_results()

    dims = ["Valence", "Arousal", "Dominance"]
    rows = []
    for dim in dims:
        a = taskA.get(dim) or {}
        b = taskB.get(dim) or {}
        rows.append({
            "Dimension": dim,
            "MAE Task A": a.get("MAE"),
            "Pearson r Task A": a.get("Pearson r"),
            "MAE Task B": b.get("MAE"),
            "Pearson r Task B": b.get("Pearson r"),
        })

    df = pd.DataFrame(rows)
    print("\nTask A vs Task B comparison:\n")
    print(df.to_string(index=False))

    df_plot = df.copy()
    metric_cols = ["MAE Task A", "MAE Task B", "Pearson r Task A", "Pearson r Task B"]
    for c in metric_cols:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    df_plot = df_plot.dropna(how="all", subset=metric_cols)

    if len(df_plot) and df_plot[metric_cols].notna().any().any():
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        df_plot.plot(x="Dimension", y=["MAE Task A", "MAE Task B"], kind="bar", ax=ax[0])
        ax[0].set_title("MAE comparison (Task A vs Task B)")
        ax[0].set_ylabel("MAE")
        df_plot.plot(x="Dimension", y=["Pearson r Task A", "Pearson r Task B"], kind="bar", ax=ax[1])
        ax[1].set_title("Pearson r comparison (Task A vs Task B)")
        ax[1].set_ylabel("Pearson r")
        plt.tight_layout()

        out = TASKB_ROOT / "risultati" / "comparativa_AB.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Comparison chart saved to", out)
    else:
        print("No numeric data to plot. Run evaluation for Task A and Task B first.")


if __name__ == "__main__":
    main()
