"""Report endpoint distributions and training-only positive class weights."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "class_distribution_report.csv"
ENDPOINTS = ("HIA", "HOB20", "HOB50")
SPLITS = ("train", "validation", "test")


def calculate_class_distribution(
    dataset_path: Path = DEFAULT_DATASET,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate split-wise counts; derive positive weights from train only."""
    data = pd.read_csv(dataset_path)
    manifest = pd.read_csv(manifest_path)
    if data["InChIKey"].duplicated().any() or manifest["InChIKey"].duplicated().any():
        raise ValueError("Dataset and split manifest must each have unique InChIKeys.")
    if set(data["InChIKey"]) != set(manifest["InChIKey"]):
        raise ValueError("Split manifest and cleaned dataset contain different InChIKey sets.")
    merged = data.merge(manifest[["InChIKey", "Split"]], on="InChIKey", how="left", validate="one_to_one")

    report_rows: list[dict[str, Any]] = []
    train_weights: dict[str, float | None] = {}
    train_counts: dict[str, dict[str, int]] = {}
    for endpoint in ENDPOINTS:
        if endpoint not in merged.columns:
            raise ValueError(f"Missing endpoint column: {endpoint}")
        for split_name in SPLITS:
            subset = merged.loc[merged["Split"] == split_name, endpoint]
            n_positive = int((subset == 1).sum())
            n_negative = int((subset == 0).sum())
            n_missing = int(subset.isna().sum())
            observed = n_positive + n_negative
            positive_pct = 100.0 * n_positive / observed if observed else float("nan")
            negative_pct = 100.0 * n_negative / observed if observed else float("nan")
            weight: float | None = None
            status = "reported; weight calculated from training split only"
            if split_name == "train":
                if n_positive == 0:
                    status = "undefined: no positive training labels"
                elif n_negative == 0:
                    status = "undefined: no negative training labels"
                else:
                    weight = n_negative / n_positive
                    status = "ok"
                train_weights[endpoint] = weight
                train_counts[endpoint] = {
                    "positive": n_positive,
                    "negative": n_negative,
                    "missing": n_missing,
                }
            report_rows.append({
                "Endpoint": endpoint,
                "Split": split_name,
                "Positive_Count": n_positive,
                "Negative_Count": n_negative,
                "Missing_Count": n_missing,
                "Positive_Percent_Of_Observed": positive_pct,
                "Negative_Percent_Of_Observed": negative_pct,
                "Train_Pos_Weight": weight if split_name == "train" else float("nan"),
                "Weight_Status": status,
            })
    report = pd.DataFrame(report_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    stats = {
        "weights": train_weights,
        "training_counts": train_counts,
        "report_path": str(output_path),
        "weights_formula": "number of observed negative training labels / number of observed positive training labels",
    }
    for endpoint in ENDPOINTS:
        logging.info(
            "%s training labels: positive=%d, negative=%d, missing=%d, pos_weight=%s",
            endpoint,
            train_counts[endpoint]["positive"],
            train_counts[endpoint]["negative"],
            train_counts[endpoint]["missing"],
            "undefined" if train_weights[endpoint] is None else f"{train_weights[endpoint]:.6f}",
        )
    logging.info("Saved split-wise distributions: %s", output_path)
    return report, stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    calculate_class_distribution(args.dataset.resolve(), args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
