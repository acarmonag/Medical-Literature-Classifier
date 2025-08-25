# def/src/model/labelwise_attention.py
# Label-Wise Attention Classifier for multi-label biomedical text classification.

import torch
import torch.nn as nn
from transformers import AutoModel

class LabelWiseAttentionClassifier(nn.Module):
    def __init__(
        self,
        encoder_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_labels: int = 4,
        att_dim: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Linear projection before attention
        self.project = nn.Linear(hidden_size, att_dim)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        # Label-wise attention vectors
        self.attention = nn.Parameter(torch.Tensor(num_labels, att_dim))
        nn.init.xavier_uniform_(self.attention)

        # Final classifier
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)

        projected = self.tanh(self.project(hidden_states))  # (B, T, A)
        att_scores = torch.einsum("bta,la->btl", projected, self.attention)  # (B, T, L)
        att_weights = torch.softmax(att_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)

        # Apply attention weights
        attended = torch.einsum("btl,bth->blh", att_weights, hidden_states)  # (B, L, H)
        logits = self.output(attended).squeeze(-1) # (B, L)

        return logits  # raw logits for BCEWithLogits
