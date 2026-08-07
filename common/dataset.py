import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmoITADataset(Dataset):
    """Wraps an EmoITA CSV file (text + optional VAD labels) for BERT.

    Used by both Task A (single target column, e.g. only 'V') and Task B
    (multiple target columns, e.g. ['V', 'A', 'D']). If target_cols is None
    or the columns are missing from the CSV, the dataset is treated as
    unlabeled (prediction mode).
    """

    def __init__(self, csv_path, tokenizer_name, max_len, target_cols=None):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.target_cols = list(target_cols) if target_cols else []

        if "text" not in self.data.columns:
            raise ValueError("CSV file must contain a 'text' column.")

        if self.target_cols and all(c in self.data.columns for c in self.target_cols):
            self.data = self.data.dropna(subset=["text", *self.target_cols]).reset_index(drop=True)
            self.data[self.target_cols] = self.data[self.target_cols].astype(float)
            self.has_labels = True
        else:
            self.data = self.data.dropna(subset=["text"]).reset_index(drop=True)
            self.has_labels = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        )
        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        }
        if self.has_labels:
            values = [row[c] / 5.0 for c in self.target_cols]
            label = values[0] if len(values) == 1 else values
            item["labels"] = torch.tensor(label, dtype=torch.float)
        return item
