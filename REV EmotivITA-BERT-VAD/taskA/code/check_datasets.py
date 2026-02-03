"""
Script di debug per controllare rapidamente la struttura dei CSV in taskA/data.
Non viene usato nel training: serve solo per ispezionare colonne, numero righe e tipi.
"""

import glob

import pandas as pd

# Lista dei file da controllare
files = glob.glob("taskA/data/*.csv")

for f in files:
    print(f"\n--- Controllo file:", f)
    try:
        df = pd.read_csv(f)
        print("Colonne:", list(df.columns))
        print("Numero righe:", len(df))
        print(df.head(3))  # prime 3 righe 

        # Check colonne
        if "text" not in df.columns:
            print("Manca la colonna 'text'")
        if len(df.columns) != 2:
            print("Ci sono più di 2 colonne, dovrebbe essere solo text + target")

        # Check valori numerici nella colonna target (prima colonna diversa da 'text')
        target_cols = [c for c in df.columns if c != "text"]
        if not target_cols:
            print("Nessuna colonna target trovata (oltre a 'text').")
            continue

        target_col = target_cols[0]
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            print(f"La colonna {target_col} non è numerica")
        else:
            print(f"Colonna {target_col} è numerica")

    except Exception as e:
        print("Errore nel file:", e)
