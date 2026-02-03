import os
import torch  # type: ignore
from torch import nn, optim  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import pandas as pd
import yaml
from transformers import AutoTokenizer  # type: ignore

from _paths import path
from model import BertForRegression

# la classe emoita_dataset trasforma CSV in batch pronti per BERT
class EmoITA_Dataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, max_len):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        if 'text' not in self.data.columns:
            raise ValueError("Il file CSV deve contenere una colonna 'text'.")

        if all(col in self.data.columns for col in ['V','A','D']):
            self.data.dropna(subset=['text','V','A','D'], inplace=True)
            self.data[['V','A','D']] = self.data[['V','A','D']].astype(float)
            self.has_labels = True
        else:
            self.data.dropna(subset=['text'], inplace=True)
            self.has_labels = False

    def __len__(self):
        return len(self.data)

    # ritorna un dizionario con input_ids e attention_mask
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
            # input_ids contiene encoding del testo
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            # attention_mask ha lo scopo di ignorare i token di padding
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long)
        }
        if self.has_labels:
            item['labels'] = torch.tensor([
                row['V']/5.0, row['A']/5.0, row['D']/5.0 # normalizzazione
            ], dtype=torch.float)
        return item

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

        total_loss += loss.item()

        # stampa ogni 50 batch
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


# eval_epoch serve per valutare il modello
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
    # carica config da taskB/config.yaml
    config_path = path("config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # path assoluti per dataset (sotto taskB/)
    tokenizer_name = cfg["tokenizer"]["name"]
    max_len = cfg["tokenizer"]["max_len"]

    train_dataset = EmoITA_Dataset(path(cfg["dataset"]["train"]), tokenizer_name, max_len)
    val_dataset   = EmoITA_Dataset(path(cfg["dataset"]["val"]), tokenizer_name, max_len)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg["params"]["batch_size"],
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=cfg["params"]["batch_size"],
                              shuffle=False)

    # modello (config_path per caricare BERT e path relativi nel config)
    model = BertForRegression(config_path=config_path)
    model.to(device)

    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["params"]["lr"])

    num_epochs = cfg["params"]["num_epochs"]
    patience = 3  # early stopping: se non migliora per 3 epoche, si ferma
    best_val_loss = float("inf")
    patience_counter = 0

    # output in taskB/risultati/
    os.makedirs(path("risultati"), exist_ok=True)
    train_losses_path = path("risultati", "train_losses.txt")
    val_losses_path = path("risultati", "val_losses.txt")
    model_path = path("risultati", "model.pth")

    open(train_losses_path, "w").close()
    open(val_losses_path, "w").close()

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # salva train e val loss per ogni epoca
        with open(train_losses_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f}\n")
        with open(val_losses_path, "a") as f:
            f.write(f"{epoch+1},{val_loss:.4f}\n")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print("Miglior modello salvato (val loss più bassa).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping attivato: nessun miglioramento su validation.")
                break

    print("\nTraining completato.")

if __name__ == "__main__":
    main()
