"""
Definizione modello Transformer per la regressione multi-target su EmoITA.

- Si usa BERT pre-addestrato come base.
- Aggiungiamo un livello fully connected linear per predire i valori continui
  di Valence (V), Arousal (A), e Dominance (D).
"""

import torch # type: ignore
from torch import nn # type: ignore
from transformers import AutoModel # type: ignore
import yaml

class BertForRegression(nn.Module):
    def __init__(self, config_path=None, output_dim=3):
        super(BertForRegression, self).__init__()
        if config_path is None:
            import os
            _dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(_dir), "config.yaml")

        # carica nome modello da config.yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["tokenizer"]["name"]

        # modello BERT pre-addestrato
        self.bert = AutoModel.from_pretrained(model_name)

        # Congeliamo i primi 10 layer di BERT in modo robusto
        for name, param in self.bert.named_parameters():
            # caso tipico: encoder.layer.<i>....
            if "encoder.layer" in name:
                try:
                    layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_idx < 10:
                        param.requires_grad = False
                except Exception:
                    # fallback: non congeliamo se non si riesce a fare parsing
                    pass
            # compatibilità con altri naming (es. 'layer.' presente)
            elif "layer." in name:
                parts = name.split(".")
                # cerca il primo token numerico tra le parti
                for p in parts:
                    if p.isdigit():
                        try:
                            if int(p) < 10:
                                param.requires_grad = False
                        except Exception:
                            pass
                        break

        self.dropout = nn.Dropout(p=0.3)  # per ridurre overfitting
        self.regressor = nn.Linear(self.bert.config.hidden_size, output_dim)  # livello lineare che predice VAD

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Usiamo il token [CLS] come rappresentazione della frase
        # compatibilità con diverse versioni di transformers
        if hasattr(outputs, "last_hidden_state"):
            seq_output = outputs.last_hidden_state
        else:
            seq_output = outputs[0]
        cls_output = seq_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.regressor(cls_output)
        return out