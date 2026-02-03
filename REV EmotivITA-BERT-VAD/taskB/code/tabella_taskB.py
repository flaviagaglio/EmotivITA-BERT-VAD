"""
Stampa la tabella dei risultati Task B leggendo da taskB/risultati/risultati.txt.
Eseguibile da qualsiasi directory.
"""
import re
from pathlib import Path

import pandas as pd

from _paths import path, TASKB_ROOT


def main():
    p = Path(TASKB_ROOT) / "risultati" / "risultati.txt"
    if not p.exists():
        print("File risultati non trovato:", p)
        print("Eseguire prima: python taskB/code/evaluation.py")
        return

    text = p.read_text(encoding="utf-8")
    dim_map = {"V": "Valence", "A": "Arousal", "D": "Dominance"}
    rows = []
    current_dim = None
    mae = None
    corr = None

    for line in text.splitlines():
        line = line.strip()
        if line in ("V:", "A:", "D:"):
            if current_dim is not None and mae is not None:
                rows.append({
                    "Dimensione": dim_map[current_dim],
                    "MAE": mae,
                    "Pearson r": corr,
                })
            current_dim = line.rstrip(":")
            mae = None
            corr = None
            continue
        if current_dim and "MAE" in line:
            m = re.search(r"MAE\s*=\s*([\d.]+)", line)
            if m:
                mae = float(m.group(1))
        if current_dim and "Pearson" in line:
            m = re.search(r"[\d.]+", line.split("=")[-1].strip())
            if m:
                corr = float(m.group(0))

    if current_dim is not None and mae is not None:
        rows.append({
            "Dimensione": dim_map[current_dim],
            "MAE": mae,
            "Pearson r": corr,
        })

    if not rows:
        print("Nessun risultato parsato da risultati.txt")
        return

    df = pd.DataFrame(rows)
    print("\nTabella Task B:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
