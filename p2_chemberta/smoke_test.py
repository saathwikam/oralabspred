from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from data import ENDPOINTS, load_p1_data
from model import ChemBERTaMultiTaskClassifier, DEFAULT_CHECKPOINT


P1_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint-backed P2 tokenization/forward/loss smoke test.")
    parser.add_argument("--p1-dir", type=Path, default=P1_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    p1_dir = args.p1_dir.resolve()
    if str(p1_dir) not in sys.path:
        sys.path.insert(0, str(p1_dir))
    from masked_weighted_loss import MaskedWeightedBCELoss

    frame, pos_weights = load_p1_data(p1_dir)
    sample = frame.head(2).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    encoded = tokenizer(
        sample["Canonical_SMILES"].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoder = AutoModel.from_pretrained(args.checkpoint)
    model = ChemBERTaMultiTaskClassifier(encoder).eval()
    with torch.inference_mode():
        logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    labels = torch.as_tensor(sample[list(ENDPOINTS)].to_numpy(dtype="float32"), dtype=torch.float32)
    criterion = MaskedWeightedBCELoss([pos_weights[name] for name in ENDPOINTS])
    loss = criterion(logits, labels)
    assert encoded["input_ids"].shape == (2, args.max_length)
    assert logits.shape == (2, 3)
    assert torch.isfinite(loss)
    assert torch.isnan(labels).any(), "Smoke-test sample should exercise the missing-label mask."
    print(f"PASS: tokenization={tuple(encoded['input_ids'].shape)}, logits={tuple(logits.shape)}, masked_loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
