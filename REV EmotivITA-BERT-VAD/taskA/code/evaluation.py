"""
Valuta le predizioni del modello EmoITA (Task A, single-target).
Confronta le predizioni con le etichette gold per una sola dimensione (V, A o D).
Percorsi e parametri sono letti dal config YAML.
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error # type: ignore
from scipy.stats import pearsonr # type: ignore
import yaml
import sys

def main():
    # leggi config da riga di comando
    if len(sys.argv) < 3:
        print("Uso: python evaluation.py --config taskA/config_valence.yaml")
        return
    config_path = sys.argv[2]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # percorso predictions
    preds_path = cfg["model_save_path"].replace(".pth", "_pred.csv")
    # percorso test gold labels
    gold_path = cfg["testset"]
    # colonna V, A o D
    target_col = cfg["target_col"]

    # carica file
    preds = pd.read_csv(preds_path)  # predizioni del modello
    gold = pd.read_csv(gold_path)    # gold labels di EmoITA

    # allinea lunghezze: prendi solo le prime N righe del gold
    # per "sincronizzazione"
    n = len(preds)
    gold = gold.iloc[:n].reset_index(drop=True)
    preds = preds.reset_index(drop=True)

    # apri file per scrivere risultati
    out_path = cfg["model_save_path"].replace(".pth", "_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Valutazione del modello EmoITA - {target_col}\n")
        f.write("===========================================\n\n")

        # calcolo metriche ufficiali (MAE e Pearson r)
        # forza array 1D
        y_true = gold[target_col].values.ravel()                # valori gold
        y_pred = preds[f"{target_col}_pred"].values.ravel()     # predizioni del modello

        mae = mean_absolute_error(y_true, y_pred)
        corr, _ = pearsonr(y_true, y_pred)

        f.write(f"{target_col}:\n")
        f.write(f"  MAE = {mae:.4f}\n")
        f.write(f"  Pearson’s r = {corr:.4f}\n\n")

    print(f"Valutazione completata. Risultati salvati in {out_path}")

if __name__ == "__main__":
    main()

