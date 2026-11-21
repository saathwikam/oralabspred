from __future__ import annotations

import torch
from torch import nn

ENDPOINTS = ("HIA", "HOB20", "HOB50")


class ChemBERTaDescriptorFusionClassifier(nn.Module):
    """ChemBERTa CLS representation late-fused with a descriptor MLP."""
    def __init__(self, encoder: nn.Module, descriptor_count: int = 6,
                 descriptor_hidden_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        self.descriptor_network = nn.Sequential(nn.Linear(descriptor_count, descriptor_hidden_size),
                                                nn.ReLU(), nn.Dropout(dropout))
        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier_heads = nn.ModuleList(
            nn.Linear(hidden + descriptor_hidden_size, 1) for _ in ENDPOINTS)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 2 or descriptors.shape[0] != input_ids.shape[0]:
            raise ValueError("descriptors must have shape [batch_size, descriptor_count]")
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = encoded.last_hidden_state[:, 0, :]
        desc = self.descriptor_network(descriptors)
        fused = self.fusion_dropout(torch.cat((cls, desc), dim=1))
        return torch.cat([head(fused) for head in self.classifier_heads], dim=1)
