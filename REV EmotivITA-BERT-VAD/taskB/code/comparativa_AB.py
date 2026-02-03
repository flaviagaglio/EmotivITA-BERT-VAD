"""
Confronto Task A vs Task B: legge i risultati da evaluation (Task A: taskA/risultati/*_results.txt,
Task B: taskB/risultati/risultati.txt) e produce tabella + grafico.
Eseguibile da qualsiasi directory.
"""
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

from _paths import path, TASKB_ROOT


def parse_taskA_results():
    """Legge MAE e Pearson r per V, A, D dai file model_*_results.txt in taskA/risultati/."""
    dim_files = [
        ("Valence", "model_valence_results.txt"),
        ("Arousal", "model_arousal_results.txt"),
        ("Dominance", "model_dominance_results.txt"),
    ]
    # taskA è sibling di taskB
    taska_root = Path(TASKB_ROOT).parent / "taskA" / "risultati"
    result = {}
    for dim, fname in dim_files:
        p = taska_root / fname
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
                m = re.search(r"[\d.]+", line.split("=")[-1])
                if m:
                    corr = float(m.group(0))
        result[dim] = {"MAE": mae, "Pearson r": corr}
    return result


def parse_taskB_results():
    """Legge MAE e Pearson r per V, A, D da taskB/risultati/risultati.txt."""
    p = Path(TASKB_ROOT) / "risultati" / "risultati.txt"
    result = {}
    if not p.exists():
        return {"Valence": {"MAE": None, "Pearson r": None},
                "Arousal": {"MAE": None, "Pearson r": None},
                "Dominance": {"MAE": None, "Pearson r": None}}
    text = p.read_text(encoding="utf-8")
    dim = None
    for line in text.splitlines():
        line = line.strip()
        if line == "V:" or line == "A:" or line == "D:":
            dim = "Valence" if line == "V:" else "Arousal" if line == "A:" else "Dominance"
            result[dim] = {"MAE": None, "Pearson r": None}
            continue
        if dim and "MAE" in line:
            m = re.search(r"MAE\s*=\s*([\d.]+)", line)
            if m:
                result[dim]["MAE"] = float(m.group(1))
        if dim and "Pearson" in line:
            m = re.search(r"[\d.]+", line.split("=")[-1].strip())
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
            "Dimensione": dim,
            "MAE Task A": a.get("MAE"),
            "Pearson r Task A": a.get("Pearson r"),
            "MAE Task B": b.get("MAE"),
            "Pearson r Task B": b.get("Pearson r"),
        })

    df = pd.DataFrame(rows)

    # Stampa tabella
    print("\nTabella comparativa Task A vs Task B:\n")
    print(df.to_string(index=False))

    # Grafico (solo colonne numeriche)
    df_plot = df.copy()
    for c in ["MAE Task A", "MAE Task B", "Pearson r Task A", "Pearson r Task B"]:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    df_plot = df_plot.dropna(how="all", subset=["MAE Task A", "MAE Task B", "Pearson r Task A", "Pearson r Task B"])

    if len(df_plot) and (df_plot["MAE Task A"].notna().any() or df_plot["MAE Task B"].notna().any()):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        if df_plot["MAE Task A"].notna().any() or df_plot["MAE Task B"].notna().any():
            df_plot.plot(x="Dimensione", y=["MAE Task A", "MAE Task B"], kind="bar", ax=ax[0])
        ax[0].set_title("Confronto MAE (Task A vs Task B)")
        ax[0].set_ylabel("MAE")
        if df_plot["Pearson r Task A"].notna().any() or df_plot["Pearson r Task B"].notna().any():
            df_plot.plot(x="Dimensione", y=["Pearson r Task A", "Pearson r Task B"], kind="bar", ax=ax[1])
        ax[1].set_title("Confronto Pearson r (Task A vs Task B)")
        ax[1].set_ylabel("Pearson r")
        plt.tight_layout()
        out = path("risultati", "comparativa_AB.png")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        plt.close()
        print("Grafico comparativo salvato in", out)
    else:
        print("Nessun dato numerico per il grafico. Eseguire prima evaluation per Task A e Task B.")


if __name__ == "__main__":
    main()