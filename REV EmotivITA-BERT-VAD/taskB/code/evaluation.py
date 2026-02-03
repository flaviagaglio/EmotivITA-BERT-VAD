"""
Valuta le predizioni del modello EmoITA confrontandole con le etichette gold.
Percorsi e parametri sono letti da config.yaml. Eseguibile da qualsiasi directory.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import yaml
import sys

from _paths import path

# anticrash per pearson per metriche robuste
def safe_pearsonr(x, y):
    try:
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return 0.0  # correlazione non definita per vettori costanti
        return pearsonr(x, y)[0]
    except Exception:
        return 0.0


def main():
    # carica config da taskB/config.yaml (o path passato da argv)
    cfg_path = path("config.yaml")
    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    preds_path = path(cfg.get("predictions_path", "risultati/my_predictions.csv"))
    gold_path = path(cfg["dataset"]["gold"])
    out_path = path(cfg.get("results_path", "risultati/risultati.txt"))

    # carica file
    preds = pd.read_csv(preds_path)
    gold = pd.read_csv(gold_path)

    n_preds, n_gold = len(preds), len(gold)

    # preferenza merge su 'id', poi su 'text', altrimenti allinea per posizione
    if 'id' in preds.columns and 'id' in gold.columns:
        merged = pd.merge(preds, gold, on="id", how="inner", suffixes=('_pred', '_gold'))
    elif 'text' in preds.columns and 'text' in gold.columns:
        merged = pd.merge(preds, gold, on="text", how="inner", suffixes=('_pred', '_gold'))
    else:
        # fallback: allinea per posizione solo se le lunghezze coincidono
        if len(preds) != len(gold):
            raise AssertionError(
                "Mismatch lunghezze pred/gold e nessuna colonna comune per il merge. "
                f"Pred: {len(preds)}, Gold: {len(gold)}"
            )
        merged = pd.concat([preds.reset_index(drop=True), gold.reset_index(drop=True)], axis=1)

    n_merged = len(merged)
    # Controllo allineamento: avvisa se molte righe sono state perse nel merge
    if n_merged < n_preds or n_merged < n_gold:
        print(
            f"[ATTENZIONE] Merge: {n_merged} righe allineate (pred: {n_preds}, gold: {n_gold}). "
            "Verificare che test set e gold labels contengano le stesse istanze (stesso id o text)."
        )
    if n_merged == 0:
        raise ValueError("Nessuna riga in comune tra predizioni e gold. Controllare i file.")

    # apri file per scrivere risultati
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Valutazione del modello EmoITA\n")
        f.write("===============================\n\n")

        for dim in ['V', 'A', 'D']:
            # trova colonne gold e pred nel merged
            if f"{dim}_pred" not in merged.columns:
                raise ValueError(f"Colonna predizione mancante: {dim}_pred")
            if dim in merged.columns:
                y_true = merged[dim].values
            elif f"{dim}_gold" in merged.columns:
                y_true = merged[f"{dim}_gold"].values
            else:
                raise ValueError(f"Colonna gold mancante per dimensione: {dim}")

            y_pred = merged[f"{dim}_pred"].values

            # se le predizioni sono nella scala [0,1], denormalizza automaticamente a [1,5]
            # solo per sicurezza
            if np.nanmax(y_pred) <= 1.0 + 1e-8:
                y_pred = y_pred * 5.0

            mae = mean_absolute_error(y_true, y_pred)
            corr = safe_pearsonr(y_true, y_pred)

            f.write(f"{dim}:\n")
            f.write(f"  MAE = {mae:.4f}\n")
            f.write(f"  Pearson’s r = {corr:.4f}\n\n")

    print(f"Valutazione completata. Risultati salvati in '{out_path}'")

if __name__ == "__main__":
    main()