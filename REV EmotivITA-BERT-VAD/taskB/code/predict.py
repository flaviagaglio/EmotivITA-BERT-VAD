"""
Per usare il modello già addestrato (model.pth) per fare predizioni
su un insieme di frasi contenute nel file test.
Percorsi e parametri sono letti da config.yaml. Eseguibile da qualsiasi directory.
"""

import os
import torch  # type: ignore
import pandas as pd
import yaml
from torch.utils.data import Dataset, DataLoader  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from _paths import path
from model import BertForRegression

# Dataset personalizzato (stesso schema di train.py)
class EmoITA_Dataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, max_len):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        if 'text' not in self.data.columns:
            raise ValueError("Il file CSV deve contenere una colonna 'text'.")

        # se non ci sono etichette, è solo test
        self.data.dropna(subset=['text'], inplace=True)
        self.has_labels = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length'
        )
        item = {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long)
        }
        return item

def predict(model, data_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs * 5.0  # denormalizza le predizioni VAD

            if i == 0:
                print("Output shape:", outputs.shape)

            preds.extend(outputs.cpu().numpy())
    return preds

def main():
    # carica config da taskB/config.yaml
    with open(path("config.yaml")) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # path assoluti
    test_csv = path(cfg["dataset"]["test"])
    out_csv = path(cfg.get("predictions_path", "risultati/my_predictions.csv"))
    model_path = path("risultati", "model.pth")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    test_dataset = EmoITA_Dataset(test_csv,
                                  cfg["tokenizer"]["name"],
                                  cfg["tokenizer"]["max_len"])
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg["params"]["batch_size"],
                             shuffle=False)

    print("Numero di frasi nel test set:", len(test_dataset))
    if len(test_dataset) == 0:
        print("Il test set è vuoto o malformato.")
        return

    # carica modello
    model = BertForRegression(config_path=path("config.yaml"))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # predizioni
    predictions = predict(model, test_loader, device)

    # salva CSV con predizioni
    df_test = pd.read_csv(test_csv).reset_index(drop=True)
    df_preds = pd.DataFrame(predictions, columns=['V_pred', 'A_pred', 'D_pred'])
    df_out = pd.concat([df_test, df_preds], axis=1)
    df_out.to_csv(out_csv, index=False)
    print("Predizioni salvate in", out_csv)

if __name__ == "__main__":
    main()
