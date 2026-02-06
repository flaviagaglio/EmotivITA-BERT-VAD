import torch # type: ignore
from torch import nn, optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import pandas as pd
import yaml
import sys
from transformers import AutoTokenizer # type: ignore
from model import BertForSingleRegression   # usa il modello single-target

# la classe trasforma CSV in batch pronti per BERT (ponte tra CSV e Pytorch)
class EmoITA_Dataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, max_len, target_col=None):
        # legge csv
        self.data = pd.read_csv(csv_path)
        # tokenizza
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.target_col = target_col

        if 'text' not in self.data.columns:
            raise ValueError("Il file CSV deve contenere una colonna 'text'.")

        if target_col and target_col in self.data.columns:
            self.data.dropna(subset=['text', target_col], inplace=True)
            self.data[target_col] = self.data[target_col].astype(float)
            self.has_labels = True
        else:
            self.data.dropna(subset=['text'], inplace=True)
            self.has_labels = False


    # controlla righe csv
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
        # normalizza
        if self.has_labels:
            item['labels'] = torch.tensor(row[self.target_col]/5.0, dtype=torch.float)
        return item

# addestramento per un'epoca completa 
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() # Somma loss di QUESTO batch

        if batch_idx % 50 == 0: # OGNI 50 batch...
            print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")
            # Stampa loss SINGOLO batch (istantanea)

    return total_loss / len(data_loader) # Loss media per epoca

# testa il modello senza cambiarlo 
def eval_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader) 


def main():
    # carica config passato da riga di comando (controllo di base)
    if len(sys.argv) < 3:
        print("Uso: python train.py --config taskA/config_valence.yaml")
        return
    config_path = sys.argv[2]

    # legge il file config passato
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # sceglie cpu o gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_name = cfg["tokenizer"]["name"]
    max_len = cfg.get("tokenizer", {}).get("max_len", 128)
    target_col = cfg["target_col"]

    # crea datasets
    train_dataset = EmoITA_Dataset(cfg["dataset"], tokenizer_name, max_len, target_col)
    val_dataset   = EmoITA_Dataset(cfg["testset"], tokenizer_name, max_len, target_col)

    #crea dataloader (batch)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=False)

    #modulo bert
    model = BertForSingleRegression(config_path=config_path)
    model.to(device)


    #def loss+opt
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    num_epochs = cfg["training"]["epochs"]
    patience = 3
    best_val_loss = float("inf")
    patience_counter = 0

    # loop epoche + early stopping
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # SALVATAGGIO MIGLIOR MODELLO
            torch.save(model.state_dict(), cfg["model_save_path"])
            print("Miglior modello salvato (val loss più bassa).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping attivato.")
                break

    print("\nTraining completato.")


if __name__ == "__main__":
    main()

