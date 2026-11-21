"""Local, deterministic summaries of MolInsight prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

EXPLANATION_ENGINE_VERSION = "deterministic-v2"


@dataclass(frozen=True)
class PredictionExplanation:
    text: str
    compound_name: str | None
    smiles: str

    def as_text(self) -> str:
        return self.text


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _label(predictions: dict[str, Any], endpoint: str) -> int | None:
    result = predictions.get(endpoint)
    if not isinstance(result, dict):
        return None
    value = result.get("prediction")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    return None


def _descriptor_sentences(descriptors: dict[str, Any] | None) -> str:
    values = descriptors if isinstance(descriptors, dict) else {}
    factors: list[tuple[int, str]] = []
    mw = _number(values.get("Molecular weight (g/mol)"))
    tpsa = _number(values.get("Topological polar surface area (Å²)"))
    hbd = _number(values.get("Hydrogen bond donors"))
    hba = _number(values.get("Hydrogen bond acceptors"))
    logp = _number(values.get("XLogP"))
    rotors = _number(values.get("Rotatable bonds"))
    if mw is not None and mw > 500:
        factors.append((1, f"its molecular weight ({mw:g} g/mol) is relatively high, which may make passive membrane passage more difficult"))
    if tpsa is not None and tpsa > 140:
        factors.append((1, f"its polar surface area ({tpsa:g} Å²) is relatively high and may limit passive permeability"))
    if (hbd is not None and hbd > 5) or (hba is not None and hba > 10):
        parts = []
        if hbd is not None:
            parts.append(f"{hbd:g} hydrogen-bond donors")
        if hba is not None:
            parts.append(f"{hba:g} acceptors")
        factors.append((1, f"it has {' and '.join(parts)}, contributing to polarity that may affect membrane passage"))
    if logp is not None and logp > 5:
        factors.append((2, f"its calculated lipophilicity (XLogP {logp:g}) is very high, which may affect aqueous solubility"))
    if rotors is not None and rotors > 10:
        factors.append((3, f"it has {rotors:g} rotatable bonds, so its flexibility may influence permeation"))
    factors.sort(key=lambda item: item[0])
    selected = [factor for _, factor in factors[:3]]
    if not selected:
        available = []
        if mw is not None:
            available.append(f"molecular weight {mw:g} g/mol")
        if tpsa is not None:
            available.append(f"polar surface area {tpsa:g} Å²")
        if hbd is not None:
            available.append(f"{hbd:g} hydrogen-bond donors")
        if hba is not None:
            available.append(f"{hba:g} hydrogen-bond acceptors")
        if logp is not None and logp > 5:
            available.append(f"XLogP {logp:g}")
        if available:
            return "Available descriptors include " + ", ".join(available[:3]) + ", but they do not establish why the model produced these results."
        return "No usable molecular descriptors were available to suggest a specific property pattern."
    return "The descriptor profile suggests that " + "; and ".join(selected) + "."


def generate_prediction_explanation(
    *, compound_name: str | None, smiles: str, predictions: dict[str, Any], descriptors: dict[str, Any] | None
) -> PredictionExplanation:
    """Return a compact three-sentence explanation without an external model."""
    name = compound_name.strip() if isinstance(compound_name, str) and compound_name.strip() else None
    subject = name or "this compound"
    labels = {endpoint: _label(predictions or {}, endpoint) for endpoint in ("HIA", "HOB20", "HOB50")}
    hia, hob20, hob50 = labels["HIA"], labels["HOB20"], labels["HOB50"]
    if all(value == 0 for value in labels.values()):
        summary = f"For {subject}, the model predicts low intestinal absorption and unfavorable oral bioavailability."
    elif all(value == 1 for value in labels.values()):
        summary = f"For {subject}, the model predicts favorable intestinal absorption and oral bioavailability."
    elif hia is not None and hob20 is not None and hob50 is not None:
        summary = (
            f"For {subject}, the model predicts {'favorable' if hia else 'low'} intestinal absorption, "
            f"with {'favorable' if hob20 else 'unfavorable'} HOB20 and "
            f"{'favorable' if hob50 else 'unfavorable'} HOB50 classifications."
        )
    else:
        available = [f"{key} {'favorable' if value else 'unfavorable'}" for key, value in labels.items() if value is not None]
        summary = f"For {subject}, the available model predictions are " + (", ".join(available) if available else "incomplete") + "; predictions are incomplete because some outcomes are unavailable."
    discrepancy = hia == 1 and (hob20 == 0 or hob50 == 0)
    distinction = (
        "Favorable intestinal absorption does not guarantee favorable oral bioavailability, which may also be affected by factors such as first-pass metabolism."
        if discrepancy else
        "Intestinal absorption and oral bioavailability are related but distinct model outcomes."
    )
    text = f"{summary} {_descriptor_sentences(descriptors)} {distinction} These properties may contribute to the predictions but are not confirmed causes."
    return PredictionExplanation(text=text, compound_name=name, smiles=smiles or "")
