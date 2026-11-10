from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, rdBase

from model import ENDPOINTS, load_saved_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ENV_VAR = "ORAL_ABSORPTION_MODEL_DIR"


class InvalidSmilesError(ValueError):
    """Raised when the supplied input cannot be parsed as a molecular structure."""


def consistency_warning(predictions: dict[str, dict[str, Any]]) -> str | None:
    if predictions["HOB50"]["prediction"] == 1 and predictions["HOB20"]["prediction"] == 0:
        return (
            "HOB20/HOB50 consistency warning: HOB50 is predicted positive while HOB20 is "
            "predicted negative. Predictions are shown unchanged."
        )
    return None


def default_model_dir() -> Path:
    configured = os.environ.get(MODEL_ENV_VAR)
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        PROJECT_ROOT.parent
        / f"{PROJECT_ROOT.name}_Kaggle_Bundle"
        / "outputs"
        / PROJECT_ROOT.name
        / "best_model"
    )


class MolecularAbsorptionPredictor:
    """Inference wrapper around the recovered best three-head ChemBERTa model."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else default_model_dir()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Trained model directory does not exist: {self.model_dir}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.tokenizer, self.config = load_saved_model(self.model_dir, device=self.device)
        if tuple(self.config.get("endpoints", ())) != ENDPOINTS:
            raise ValueError(
                f"Model endpoints do not match the application: {self.config.get('endpoints')}"
            )
        self.max_length = int(self.config["max_length"])
        self.threshold = float(self.config.get("decision_threshold", 0.5))
        if self.max_length <= 0 or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Model configuration has an invalid max_length or decision_threshold.")

    @staticmethod
    def canonicalize_smiles(smiles: str) -> str:
        if not isinstance(smiles, str) or not smiles.strip():
            raise InvalidSmilesError("Enter a non-empty SMILES string.")
        try:
            # RDKit logs parse diagnostics to stderr; the UI reports a concise error instead.
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles.strip())
            if molecule is None:
                raise InvalidSmilesError("RDKit could not parse this SMILES string.")
            return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        except InvalidSmilesError:
            raise
        except Exception as exc:
            raise InvalidSmilesError(f"SMILES validation failed: {exc}") from exc

    def predict(self, smiles: str) -> dict[str, Any]:
        canonical_smiles = self.canonicalize_smiles(smiles)
        encoded = self.tokenizer(
            [canonical_smiles],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
            )
            probabilities = torch.sigmoid(logits).float().cpu().numpy()[0]
        if probabilities.shape != (len(ENDPOINTS),) or not np.isfinite(probabilities).all():
            raise RuntimeError("The trained model returned invalid prediction probabilities.")

        predictions = {
            endpoint: {
                "probability": float(probability),
                "prediction": int(probability >= self.threshold),
                "threshold": self.threshold,
            }
            for endpoint, probability in zip(ENDPOINTS, probabilities, strict=True)
        }
        return {
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "predictions": predictions,
            "consistency_warning": consistency_warning(predictions),
        }
