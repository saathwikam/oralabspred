from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ENDPOINTS = ("HIA", "HOB20", "HOB50")
DEFAULT_CHECKPOINT = "DeepChem/ChemBERTa-77M-MLM"


class ChemBERTaMultiTaskClassifier(nn.Module):
    """One ChemBERTa encoder with one independent binary head per endpoint."""

    def __init__(self, encoder: nn.Module, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier_heads = nn.ModuleList(
            nn.Linear(hidden_size, 1) for _ in ENDPOINTS
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls_embedding)
        return torch.cat([head(pooled) for head in self.classifier_heads], dim=1)


def load_saved_model(model_dir: str | Path, device: str | torch.device = "cpu"):
    """Load a saved P2 model and tokenizer for later application inference."""
    model_dir = Path(model_dir)
    with (model_dir / "model_config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    encoder = AutoModel.from_pretrained(model_dir / "encoder")
    model = ChemBERTaMultiTaskClassifier(encoder, dropout=float(config["dropout"]))
    state = torch.load(model_dir / "model_state_dict.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    return model, tokenizer, config
