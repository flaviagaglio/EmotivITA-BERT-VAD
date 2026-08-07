import warnings

from torch import nn
from transformers import AutoModel

# BERT loads with some pretraining-only weights that we don't use
# (e.g. the next-sentence-prediction head). This warning is expected.
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")


def _layer_index(param_name):
    """Extracts the transformer layer index from a BERT parameter name.
    Handles both 'encoder.layer.N....' and 'layer.N....' naming."""
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def freeze_bottom_layers(bert, n_layers):
    """Freezes the first n_layers of BERT, keeping the top layers trainable."""
    for name, param in bert.named_parameters():
        idx = _layer_index(name)
        if idx is not None and idx < n_layers:
            param.requires_grad = False


class BertVADRegressor(nn.Module):
    """BERT + linear regressor for VAD prediction.

    output_dim=1 for single-target regression (Task A: only V, A or D),
    output_dim=3 for joint regression (Task B: V, A and D together).
    """

    def __init__(self, model_name, output_dim=1, freeze_layers=10, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.bert = AutoModel.from_pretrained(model_name)
        freeze_bottom_layers(self.bert, freeze_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.regressor(cls_output)
        return out.squeeze(-1) if self.output_dim == 1 else out
