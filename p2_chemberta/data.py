from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

ENDPOINTS = ("HIA", "HOB20", "HOB50")
SPLITS = ("train", "validation", "test")


class MoleculeDataset(Dataset):
    def __init__(self, encoded: dict[str, torch.Tensor], targets: torch.Tensor) -> None:
        self.encoded = encoded
        self.targets = targets

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encoded["input_ids"][index],
            "attention_mask": self.encoded["attention_mask"][index],
            "labels": self.targets[index],
        }


def load_p1_data(p1_dir: str | Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Join existing P1 files by InChIKey without changing their contents."""
    p1_dir = Path(p1_dir)
    processed = p1_dir / "data" / "processed"
    data = pd.read_csv(processed / "cleaned_dataset.csv")
    manifest = pd.read_csv(processed / "split_manifest.csv")
    weights = pd.read_csv(processed / "class_distribution_report.csv")

    for frame, name in ((data, "cleaned dataset"), (manifest, "split manifest")):
        if "InChIKey" not in frame or frame["InChIKey"].duplicated().any():
            raise ValueError(f"{name} must have a unique InChIKey column.")
    if set(data["InChIKey"]) != set(manifest["InChIKey"]):
        raise ValueError("Cleaned dataset and frozen split manifest have different InChIKey sets.")
    required_data = {"Canonical_SMILES", *ENDPOINTS}
    missing = required_data.difference(data.columns)
    if missing:
        raise ValueError(f"Cleaned dataset is missing columns: {sorted(missing)}")

    joined = data.merge(
        manifest[["InChIKey", "Split"]], on="InChIKey", how="left", validate="one_to_one", sort=False
    )
    if joined["Split"].isna().any() or not set(joined["Split"]).issubset(SPLITS):
        raise ValueError("Split manifest contains missing or unsupported split assignments.")
    if joined["Canonical_SMILES"].isna().any() or joined["Canonical_SMILES"].astype(str).str.strip().eq("").any():
        raise ValueError("Cleaned dataset contains missing Canonical_SMILES.")

    for endpoint in ENDPOINTS:
        values = pd.to_numeric(joined[endpoint], errors="raise").to_numpy(dtype=np.float64)
        observed = values[np.isfinite(values)]
        if not np.isin(observed, (0.0, 1.0)).all() or np.isinf(values).any():
            raise ValueError(f"{endpoint} labels must be 0, 1, or missing NaN.")
        joined[endpoint] = values

    train_weights = weights.loc[weights["Split"].eq("train")].set_index("Endpoint")
    if train_weights.index.duplicated().any():
        raise ValueError("Class-weight report has duplicate training endpoint rows.")
    pos_weights: dict[str, float] = {}
    for endpoint in ENDPOINTS:
        if endpoint not in train_weights.index:
            raise ValueError(f"No training-only class weight found for {endpoint}.")
        value = train_weights.loc[endpoint, "Train_Pos_Weight"]
        if pd.isna(value) or not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Training-only class weight is undefined for {endpoint}.")
        pos_weights[endpoint] = float(value)
    return joined, pos_weights


def tokenize_frame(
    frame: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> MoleculeDataset:
    smiles = frame["Canonical_SMILES"].astype(str).tolist()
    encoded = tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    targets = torch.as_tensor(frame[list(ENDPOINTS)].to_numpy(dtype=np.float32), dtype=torch.float32)
    return MoleculeDataset(
        {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}, targets
    )
