from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DESCRIPTOR_NAMES = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds")


def calculate_descriptors(smiles: str) -> tuple[float, ...]:
    """Compute the fixed RDKit descriptor vector for one SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES is empty or not a string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError("RDKit could not parse the SMILES")
    values = (Descriptors.MolWt(molecule), Crippen.MolLogP(molecule),
              rdMolDescriptors.CalcTPSA(molecule), Lipinski.NumHDonors(molecule),
              Lipinski.NumHAcceptors(molecule), Lipinski.NumRotatableBonds(molecule))
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(DESCRIPTOR_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("RDKit returned invalid or non-finite descriptors")
    return tuple(float(x) for x in vector)


def extract_descriptor_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one descriptor row per input and an explicit failure audit; never drop rows."""
    records, failures = [], []
    for index, row in frame.reset_index(drop=True).iterrows():
        key, smiles = row.get("InChIKey"), row.get("Canonical_SMILES")
        try:
            values = calculate_descriptors(smiles)
            records.append({"InChIKey": key, **dict(zip(DESCRIPTOR_NAMES, values, strict=True))})
        except Exception as exc:
            records.append({"InChIKey": key, **{name: np.nan for name in DESCRIPTOR_NAMES}})
            failures.append({"dataset_row_index_zero_based": int(index), "InChIKey": key,
                             "Canonical_SMILES": smiles, "error_type": type(exc).__name__, "error": str(exc)})
    failure_columns = ["dataset_row_index_zero_based", "InChIKey", "Canonical_SMILES", "error_type", "error"]
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(failures, columns=failure_columns)


@dataclass
class DescriptorPreprocessor:
    """Median imputation and scaling fitted only on the training partition."""
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None

    def fit(self, train_values: np.ndarray) -> "DescriptorPreprocessor":
        matrix = np.asarray(train_values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(DESCRIPTOR_NAMES) or not len(matrix):
            raise ValueError("Training descriptors must have shape [n, 6], n > 0")
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(matrix)
        self.scaler = StandardScaler().fit(self.imputer.transform(matrix))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Preprocessor must be fit on training rows first")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(DESCRIPTOR_NAMES):
            raise ValueError("Descriptor matrix must have shape [n, 6]")
        result = self.scaler.transform(self.imputer.transform(matrix))
        if not np.isfinite(result).all():
            raise ValueError("Descriptor preprocessing produced non-finite values")
        return result.astype(np.float32)

    def parameters(self) -> dict:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Preprocessor has not been fit")
        return {"descriptor_names": list(DESCRIPTOR_NAMES), "imputation": "median",
                "imputer_statistics": self.imputer.statistics_.tolist(), "scaling": "StandardScaler",
                "scaler_mean": self.scaler.mean_.tolist(), "scaler_scale": self.scaler.scale_.tolist(),
                "fit_split": "train", "n_training_rows_fit": int(self.scaler.n_samples_seen_)}
