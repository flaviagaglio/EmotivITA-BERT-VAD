"""
Definizione modelli Transformer per EmoITA:
- BertForRegression: regressione multi-target (Task B, predice V, A, D insieme).
- BertForSingleRegression: regressione single-target (Task A, predice solo una dimensione).
"""

import warnings
import torch # type: ignore
from torch import nn # type: ignore
from transformers import AutoModel # type: ignore
import yaml

# Sopprime i warning sui parametri "UNEXPECTED" quando si carica BERT pre-addestrato
# (sono normali: BERT include layer di pre-training che non usiamo nel nostro modello)
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

# --- Multi-target (Task B) ---
class BertForRegression(nn.Module):
    def __init__(self, config_path="taskB/config.yaml", output_dim=3):
        super(BertForRegression, self).__init__()

        with open(config_path) as f:
            # legge da config 
            cfg = yaml.safe_load(f)
        model_name = cfg["tokenizer"]["name"]

        #caricamento bert italiano
        self.bert = AutoModel.from_pretrained(model_name)

        # Congela i primi 10 layer
        # catturano feature generali già apprese nel pre-training
        # addestra solo gli ultimi layer per adattarsi al task VAD
        for name, param in self.bert.named_parameters():
            if "layer." in name and int(name.split(".")[2]) < 10:
                param.requires_grad = False

        # drop 30%
        self.dropout = nn.Dropout(p=0.3)
        # ritornano 3 dimensioni
        self.regressor = nn.Linear(self.bert.config.hidden_size, output_dim)  # output VAD

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.regressor(cls_output)  # shape: (batch, 3)


# --- Single-target (Task A) ---
class BertForSingleRegression(nn.Module):
    def __init__(self, config_path="taskA/config.yaml", output_dim=1):
        super(BertForSingleRegression, self).__init__() 

        with open(config_path) as f:
            # legge da config
            cfg = yaml.safe_load(f) # bert italian cased
        model_name = cfg["tokenizer"]["name"]

        # carica bert italiano
        self.bert = AutoModel.from_pretrained(model_name)

        # Congela i primi 10 layer (12 layer totali, i primi 10 corrispondono a conoscenza generale italiano)
        for name, param in self.bert.named_parameters():
            if "layer." in name and int(name.split(".")[2]) < 10:
                param.requires_grad = False #no training (piu veloce)

        # dropout anti overfitting
        self.dropout = nn.Dropout(p=0.3)
        # regressore da 768 dim -> a 1
        self.regressor = nn.Linear(self.bert.config.hidden_size, output_dim)  # output singolo

    # FUNZIONAMENTO MODELLO
    def forward(self, input_ids, attention_mask): # batch da data loader
        # bert elabora 16 frasi 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # prende cls
        cls_output = outputs.last_hidden_state[:, 0, :]
        # fa dropout 30%
        cls_output = self.dropout(cls_output)
        # regressore -> 1
        out = self.regressor(cls_output)  # shape: (batch, 1)
        # predizioni
        return out.squeeze(-1)    # rimozione ultima dimensione per non avere errori loss       