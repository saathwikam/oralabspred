from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from descriptors import DESCRIPTOR_NAMES, DescriptorPreprocessor, extract_descriptor_frame

ENDPOINTS = ("HIA", "HOB20", "HOB50")
SPLITS = ("train", "validation", "test")


class DescriptorMoleculeDataset(Dataset):
    def __init__(self, encoded, descriptors: np.ndarray, targets: np.ndarray, inchikeys: list[str]):
        if not (len(encoded["input_ids"]) == len(descriptors) == len(targets) == len(inchikeys)):
            raise ValueError("Token, descriptor, target, and identifier counts differ")
        self.input_ids, self.attention_mask = encoded["input_ids"], encoded["attention_mask"]
        self.descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.inchikeys = inchikeys

    def __len__(self):
        return int(len(self.targets))

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index],
                "descriptors": self.descriptors[index], "labels": self.targets[index]}


def load_v2_data(p1_dir: str | Path):
    """Read P1 unchanged, validate the frozen join, calculate descriptors, retain every row."""
    root = Path(p1_dir)
    cleaned = pd.read_csv(root / "data" / "processed" / "cleaned_dataset.csv")
    manifest = pd.read_csv(root / "data" / "processed" / "split_manifest.csv")
    weights = pd.read_csv(root / "data" / "processed" / "class_distribution_report.csv")
    for table, name in ((cleaned, "cleaned dataset"), (manifest, "split manifest")):
        if "InChIKey" not in table or table.InChIKey.isna().any() or table.InChIKey.duplicated().any():
            raise ValueError(f"{name} requires complete, unique InChIKeys")
    if set(cleaned.InChIKey) != set(manifest.InChIKey):
        raise ValueError("P1 data and manifest key sets differ")
    frame = cleaned.merge(manifest[["InChIKey", "Canonical_SMILES", "Scaffold_ID", "Scaffold_Type", "Split"]],
                          on="InChIKey", suffixes=("_data", "_manifest"), how="left",
                          validate="one_to_one", sort=False)
    if not frame.Canonical_SMILES_data.eq(frame.Canonical_SMILES_manifest).all():
        raise ValueError("Data and manifest canonical SMILES differ")
    if frame.Split.isna().any() or not frame.Split.isin(SPLITS).all():
        raise ValueError("Invalid frozen split assignment")
    frame["Canonical_SMILES"] = frame.pop("Canonical_SMILES_data")
    frame.drop(columns="Canonical_SMILES_manifest", inplace=True)
    for endpoint in ENDPOINTS:
        values = pd.to_numeric(frame[endpoint], errors="raise").to_numpy(dtype=float)
        observed = values[np.isfinite(values)]
        if np.isinf(values).any() or not np.isin(observed, (0.0, 1.0)).all():
            raise ValueError(f"{endpoint} labels must be binary or missing")
        frame[endpoint] = values
    train_weights = weights.loc[weights.Split.eq("train")].set_index("Endpoint")
    pos_weights = {name: float(train_weights.loc[name, "Train_Pos_Weight"]) for name in ENDPOINTS}
    desc, failures = extract_descriptor_frame(frame[["InChIKey", "Canonical_SMILES"]])
    frame = frame.merge(desc, on="InChIKey", validate="one_to_one", sort=False)
    return frame, pos_weights, desc, failures


def fit_transform_descriptors(frame: pd.DataFrame):
    train = frame.Split.eq("train").to_numpy()
    preprocessor = DescriptorPreprocessor().fit(frame.loc[train, list(DESCRIPTOR_NAMES)].to_numpy(dtype=float))
    transformed = preprocessor.transform(frame[list(DESCRIPTOR_NAMES)].to_numpy(dtype=float))
    return transformed, preprocessor


def tokenize_with_descriptors(frame: pd.DataFrame, descriptors: np.ndarray, tokenizer, max_length: int):
    tokens = tokenizer(frame.Canonical_SMILES.astype(str).tolist(), padding="max_length", truncation=True,
                       max_length=max_length, return_attention_mask=True, return_tensors="pt")
    targets = frame[list(ENDPOINTS)].to_numpy(dtype=np.float32)
    return DescriptorMoleculeDataset(tokens, descriptors, targets, frame.InChIKey.astype(str).tolist())
