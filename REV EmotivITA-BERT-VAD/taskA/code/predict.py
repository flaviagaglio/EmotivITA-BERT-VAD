"""
Predizioni con modello single-target (Task A).
Usa BertForSingleRegression e un config YAML dedicato (es. config_valence.yaml).
"""

import torch # type: ignore
import pandas as pd
import yaml
import sys
from torch.utils.data import Dataset, DataLoader # type: ignore
from transformers import AutoTokenizer # type: ignore
from model import BertForSingleRegression

# Dataset personalizzato (solo testo, senza etichette). 
# Stavolta, a differenza di train, consideriamo solo testo per predizione
class EmoITA_Dataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, max_len):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        if 'text' not in self.data.columns:
            raise ValueError("Il file CSV deve contenere una colonna 'text'.")

        self.data.dropna(subset=['text'], inplace=True)

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

# generazione di tutte le predizioni
def predict(model, data_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs * 5.0  # denormalizza (torna alla scala originale)

            if i == 0:
                print("Output shape:", outputs.shape)

            preds.extend(outputs.cpu().numpy())
    return preds

def main():
    # leggi config da riga di comando
    if len(sys.argv) < 3:
        print("Uso: python predict.py --config taskA/config_valence.yaml")
        return
    config_path = sys.argv[2]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    test_csv = cfg["testset"]
    target_col = cfg["target_col"]
    batch_size = cfg["training"]["batch_size"]

    test_dataset = EmoITA_Dataset(test_csv,
                                  cfg["tokenizer"]["name"],
                                  cfg.get("tokenizer", {}).get("max_len", 128))
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    print("Numero di frasi nel test set:", len(test_dataset))
    if len(test_dataset) == 0:
        print("Il test set è vuoto o malformato.")
        return

    # carica modello single-target
    model = BertForSingleRegression(config_path=config_path)
    model.load_state_dict(torch.load(cfg["model_save_path"], map_location=device))
    model.to(device)

    # predizioni
    predictions = predict(model, test_loader, device)

    # salva CSV con predizioni
    df_test = pd.read_csv(test_csv).reset_index(drop=True)
    df_preds = pd.DataFrame(predictions, columns=[f"{target_col}_pred"])
    df_out = pd.concat([df_test, df_preds], axis=1)
    out_path = cfg["model_save_path"].replace(".pth", "_pred.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Predizioni salvate in {out_path}")

if __name__ == "__main__":
    main()

    '''eseguire con 
    python taskA/code/predict.py --config taskA/config_valence.yaml
    python taskA/code/predict.py --config taskA/config_arousal.yaml
    python taskA/code/predict.py --config taskA/config_dominance.yaml'''