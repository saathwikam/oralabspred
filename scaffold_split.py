"""Create a reproducible group split by Bemis-Murcko scaffold.

Run after ``data_preprocessing.py`` with ``python scaffold_split.py``. All
records sharing a scaffold stay together. Acyclic molecules, which have no
Bemis-Murcko scaffold, use an explicit molecule-specific fallback group.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)


def scaffold_for(smiles: str, inchi_key: str) -> tuple[str, str]:
    """Return (scaffold ID, scaffold kind); acyclic molecules get unique groups."""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"Invalid SMILES in cleaned dataset for {inchi_key}: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold, "Bemis-Murcko"
    return f"ACYCLIC::{inchi_key}", "Acyclic molecule-specific fallback"


def assign_scaffold_groups(
    group_sizes: dict[str, int], fractions: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """Greedily assign whole scaffold groups toward target row proportions."""
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("Exactly three positive split fractions are required.")
    total_fraction = sum(fractions)
    fractions = tuple(value / total_fraction for value in fractions)
    rng = random.Random(seed)
    groups = list(group_sizes.items())
    rng.shuffle(groups)  # reproducible randomized ordering for equal-size scaffolds
    groups.sort(key=lambda item: item[1], reverse=True)
    total_rows = sum(group_sizes.values())
    targets = [total_rows * fraction for fraction in fractions]
    current = [0, 0, 0]
    assignment: dict[str, str] = {}
    for scaffold_id, size in groups:
        deficits = [
            (targets[index] - current[index]) / max(targets[index], 1.0)
            for index in range(3)
        ]
        chosen = max(range(3), key=lambda index: (deficits[index], -index))
        assignment[scaffold_id] = SPLIT_NAMES[chosen]
        current[chosen] += size
    if len(groups) >= 3:
        empty = [name for name in SPLIT_NAMES if name not in assignment.values()]
        if empty:
            raise RuntimeError(f"Unable to populate all splits; empty: {empty}")
    return assignment


def create_scaffold_split(
    dataset_path: Path = DEFAULT_DATASET,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the frozen molecule-to-scaffold split manifest."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found: {dataset_path}")
    data = pd.read_csv(dataset_path)
    required = {"InChIKey", "Canonical_SMILES", "SMILES"}
    absent = required.difference(data.columns)
    if absent:
        raise ValueError(f"Cleaned dataset is missing columns: {sorted(absent)}")
    if data["InChIKey"].duplicated().any():
        raise ValueError("Cleaned dataset must contain one row per computed InChIKey.")

    scaffolds = [
        scaffold_for(str(row.Canonical_SMILES), str(row.InChIKey))
        for row in data[["Canonical_SMILES", "InChIKey"]].itertuples(index=False)
    ]
    data["Scaffold_ID"] = [item[0] for item in scaffolds]
    data["Scaffold_Type"] = [item[1] for item in scaffolds]
    group_sizes = data.groupby("Scaffold_ID", sort=False).size().to_dict()
    group_assignment = assign_scaffold_groups(group_sizes, fractions, seed)
    data["Split"] = data["Scaffold_ID"].map(group_assignment)
    manifest_columns = ["InChIKey", "Canonical_SMILES", "Scaffold_ID", "Scaffold_Type", "Split"]
    manifest = data[manifest_columns].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    counts = manifest["Split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    proportions = {name: float(counts[name] / len(manifest)) if len(manifest) else 0.0 for name in SPLIT_NAMES}
    scaffold_splits = manifest.groupby("Scaffold_ID")["Split"].nunique()
    if (scaffold_splits > 1).any():
        raise AssertionError("Scaffold leakage detected after assignment.")
    stats: dict[str, Any] = {
        "seed": int(seed),
        "fractions_requested": dict(zip(SPLIT_NAMES, fractions)),
        "molecule_counts": {name: int(counts[name]) for name in SPLIT_NAMES},
        "actual_proportions": proportions,
        "scaffold_counts": {
            name: int(manifest.loc[manifest["Split"] == name, "Scaffold_ID"].nunique())
            for name in SPLIT_NAMES
        },
        "total_scaffolds": int(manifest["Scaffold_ID"].nunique()),
        "acyclic_fallback_molecules": int((manifest["Scaffold_Type"] != "Bemis-Murcko").sum()),
        "scaffold_leakage_groups": int((scaffold_splits > 1).sum()),
        "manifest_path": str(output_path),
    }
    logging.info(
        "Split counts: train=%d (%.1f%%), validation=%d (%.1f%%), test=%d (%.1f%%)",
        counts["train"], proportions["train"] * 100,
        counts["validation"], proportions["validation"] * 100,
        counts["test"], proportions["test"] * 100,
    )
    logging.info(
        "Scaffolds=%d; acyclic fallback groups=%d; leakage groups=%d; saved %s",
        stats["total_scaffolds"], stats["acyclic_fallback_molecules"],
        stats["scaffold_leakage_groups"], output_path,
    )
    return manifest, stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    create_scaffold_split(args.dataset.resolve(), args.output.resolve(), seed=args.seed)


if __name__ == "__main__":
    main()
