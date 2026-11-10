"""Run the AI-Based Oral Absorption Prediction System P1 cleaning, scaffold split, and weight-report pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import rdkit
import torch

from class_weights import calculate_class_distribution
from data_preprocessing import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, ENDPOINTS, clean_dataset
from scaffold_split import DEFAULT_FRACTIONS, create_scaffold_split


PROJECT_ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def create_quality_report(
    input_path: Path,
    output_dir: Path,
    prep: dict[str, Any],
    split: dict[str, Any],
    weight_stats: dict[str, Any],
    seed: int,
) -> Path:
    """Write a Markdown quality and reproducibility report using computed data."""
    cleaned = pd.read_csv(output_dir / "cleaned_dataset.csv")
    distribution = pd.read_csv(output_dir / "class_distribution_report.csv")
    original = pd.read_csv(input_path)
    source_hash = sha256_file(input_path)

    dist_lines = [
        "| Endpoint | Split | Positive | Negative | Missing | Positive % | Negative % | Training pos_weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distribution.itertuples(index=False):
        weight = "—" if pd.isna(row.Train_Pos_Weight) else f"{row.Train_Pos_Weight:.6f}"
        dist_lines.append(
            f"| {row.Endpoint} | {row.Split} | {int(row.Positive_Count)} | {int(row.Negative_Count)} "
            f"| {int(row.Missing_Count)} | {row.Positive_Percent_Of_Observed:.2f}% "
            f"| {row.Negative_Percent_Of_Observed:.2f}% | {weight} |"
        )

    split_lines = ["| Split | Molecules | Actual proportion | Unique scaffold groups |", "|---|---:|---:|---:|"]
    for name in ("train", "validation", "test"):
        split_lines.append(
            f"| {name} | {split['molecule_counts'][name]} | {_pct(split['actual_proportions'][name])} "
            f"| {split['scaffold_counts'][name]} |"
        )

    semantics = prep["label_semantics"]
    limitations = [
        "The model table has partially observed endpoints; blank values are unassayed, not negative, and must remain masked during training.",
        "Class weighting addresses class imbalance only; it does not correct source, assay, or chemical-domain bias.",
        "Scaffold grouping can make actual split proportions and endpoint class proportions differ from the requested fractions.",
        "The five-column model CSV has no source-row/sheet membership. This run does not exclude compounds from the source workbook's original HOB test sheets from the new scaffold split; reconcile source identities before claiming an independent external benchmark evaluation.",
        "The 65 acyclic molecules use molecule-specific fallback groups because their Bemis-Murcko scaffold is empty; identical structures remain grouped by computed InChIKey.",
    ]
    lines = [
        "# AI-Based Oral Absorption Prediction System - P1 Dataset Quality Report",
        "",
        "## Source and reproducibility",
        "",
        f"- Source CSV: `{input_path.name}`",
        f"- Source CSV SHA-256: `{source_hash}`",
        f"- Random seed: `{seed}`",
        f"- Requested split fractions: train {_pct(DEFAULT_FRACTIONS[0])}, validation {_pct(DEFAULT_FRACTIONS[1])}, test {_pct(DEFAULT_FRACTIONS[2])}",
        f"- Python: `{platform.python_version()}`",
        f"- RDKit: `{rdkit.__version__}`; pandas: `{pd.__version__}`; NumPy: `{_package_version('numpy')}`; PyTorch: `{torch.__version__}`",
        "",
        "## Dataset and structure checks",
        "",
        f"- Original dimensions: {len(original)} rows × {len(original.columns)} columns (`{', '.join(original.columns)}`).",
        f"- Valid SMILES: {prep['valid_smiles']}; invalid or missing SMILES: {prep['invalid_rows']}.",
        f"- Unique molecules after computed-InChIKey deduplication: {prep['unique_molecules']}.",
        f"- Duplicate structure groups: {prep['duplicate_groups']}; extra duplicate rows coalesced: {prep['duplicate_extra_rows']}.",
        f"- Conflicting duplicate endpoint labels: {prep['conflicting_duplicate_endpoints']}; those endpoint values were set missing and audited.",
        f"- Input/computed InChIKey mismatches: {prep['key_mismatches']}.",
        f"- Invalid records: `{Path(prep['invalid_path']).name}`; duplicate audit: `{Path(prep['duplicate_audit_path']).name}`; label conflict audit: `{Path(prep['conflict_audit_path']).name}`.",
        "",
        "### Endpoint meanings and cleaned missing counts",
        "",
        *[f"- **{name}:** {meaning}" for name, meaning in semantics.items()],
        *[f"- Missing `{endpoint}` labels in original source: {prep['missing_labels_in_source'][endpoint]}." for endpoint in ENDPOINTS],
        *[f"- Missing `{endpoint}` labels after cleaning: {int(cleaned[endpoint].isna().sum())} / {len(cleaned)}." for endpoint in ENDPOINTS],
        "",
        "## Scaffold split",
        "",
        *split_lines,
        "",
        f"- Unique scaffold groups: {split['total_scaffolds']}; acyclic fallback molecules: {split['acyclic_fallback_molecules']}.",
        f"- Scaffold leakage groups across splits: {split['scaffold_leakage_groups']}.",
        "- Molecule-level and scaffold-level split assignment is frozen in `split_manifest.csv`.",
        "",
        "## Endpoint distributions and class weights",
        "",
        "`pos_weight = N_negative / N_positive`, calculated from observed training labels only. Validation/test rows have no training weight.",
        "",
        *dist_lines,
        "",
        "## Known limitations",
        "",
        *[f"- {item}" for item in limitations],
        "",
        "## Integration interface for P2/P3/P4",
        "",
        "Use `cleaned_dataset.csv` and `split_manifest.csv`; never make a new row-wise split in a model branch. Join one-to-one on `InChIKey`. Read target columns in `HIA`, `HOB20`, `HOB50` order and preserve NaN values. Select train/validation/test using the manifest's `Split` column. Load the three training `Train_Pos_Weight` values from `class_distribution_report.csv` in endpoint order. If a training weight is undefined because a class has no observed examples, do not silently substitute a value; report that task as not trainable with class balancing.",
        "",
        "```python",
        "import pandas as pd",
        "import torch",
        "from masked_weighted_loss import MaskedWeightedBCELoss",
        "",
        "data = pd.read_csv('data/processed/cleaned_dataset.csv')",
        "manifest = pd.read_csv('data/processed/split_manifest.csv')",
        "train_ids = set(manifest.loc[manifest.Split == 'train', 'InChIKey'])",
        "train = data[data.InChIKey.isin(train_ids)].reset_index(drop=True)",
        "weight_table = pd.read_csv('data/processed/class_distribution_report.csv')",
        "weight_rows = weight_table.query(\"Split == 'train'\").set_index('Endpoint')",
        "pos_weights = [float(weight_rows.loc[name, 'Train_Pos_Weight']) for name in ('HIA', 'HOB20', 'HOB50')]",
        "criterion = MaskedWeightedBCELoss(pos_weights)",
        "targets = torch.as_tensor(train[['HIA', 'HOB20', 'HOB50']].to_numpy(dtype='float32'))  # NaN stays missing",
        "logits = model_output  # raw [batch, 3] logits; do not apply sigmoid before this loss",
        "loss, per_task = criterion(logits, targets, return_details=True)",
        "```",
        "",
        "For RDKit descriptors, fit imputation/scaling on training rows only and reuse the fitted transform for validation and test. ChemBERTa and RDKit branches must consume the same molecule order and frozen split.",
        "",
    ]
    report_path = output_dir / "data_quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_pipeline(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT_DIR, seed: int = 42) -> dict[str, Any]:
    """Run cleaning, split generation, training-only weights, and reporting."""
    output_names = (
        "cleaned_dataset.csv", "invalid_records.csv", "duplicate_audit.csv",
        "label_conflict_audit.csv", "split_manifest.csv",
        "class_distribution_report.csv", "data_quality_report.md",
    )
    resolved_input = input_path.resolve()
    collisions = [name for name in output_names if (output_dir / name).resolve() == resolved_input]
    if collisions:
        raise ValueError(
            f"Input dataset would be overwritten by pipeline output: {collisions}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    prep = clean_dataset(input_path, output_dir)
    manifest_path = output_dir / "split_manifest.csv"
    manifest, split_stats = create_scaffold_split(
        output_dir / "cleaned_dataset.csv", manifest_path, seed=seed
    )
    _, weight_stats = calculate_class_distribution(
        output_dir / "cleaned_dataset.csv",
        manifest_path,
        output_dir / "class_distribution_report.csv",
    )
    report_path = create_quality_report(input_path, output_dir, prep, split_stats, weight_stats, seed)
    return {"preprocessing": prep, "split": split_stats, "weights": weight_stats, "report_path": str(report_path)}


def main() -> None:
    """CLI entry point for a complete reproducible P1 run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_pipeline(args.input.resolve(), args.output_dir.resolve(), args.seed)
    logging.info("P1 pipeline complete; report: %s", results["report_path"])


if __name__ == "__main__":
    main()
