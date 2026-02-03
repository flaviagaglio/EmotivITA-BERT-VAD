"""
Prepara train e validation a partire dal Development set (Task B).
Split 80% train, 20% validation. Eseguibile da qualsiasi directory.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

from _paths import path

# carica Development set (path assoluto rispetto a taskB/)
df = pd.read_csv(path("data", "Development set.csv"))

# rimozione righe con valori mancanti
df = df.dropna(subset=["text", "V", "A", "D"])

# split 80% train, 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# salva in taskB/data/
os.makedirs(path("data"), exist_ok=True)
train_df.to_csv(path("data", "train.csv"), index=False)
val_df.to_csv(path("data", "val.csv"), index=False)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
