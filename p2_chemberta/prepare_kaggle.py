from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

P2_FILES = ("data.py", "model.py", "train.py")
P1_DATA_FILES = ("cleaned_dataset.csv", "split_manifest.csv", "class_distribution_report.csv")
P1_ROOT = Path(__file__).resolve().parents[1]
P2_ROOT = Path(__file__).resolve().parent


def build_bundle(p1_dir: Path, bundle_dir: Path, username: str, dataset_slug: str, kernel_slug: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Kaggle username may contain only letters, numbers, and hyphens.")
    for slug_name, slug in (("dataset", dataset_slug), ("kernel", kernel_slug)):
        if not re.fullmatch(r"[a-z0-9-]+", slug):
            raise ValueError(f"{slug_name} slug must use lowercase letters, numbers, and hyphens.")
    p1_dir = p1_dir.resolve()
    bundle_dir = bundle_dir.resolve()
    try:
        bundle_dir.relative_to(p1_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Bundle output must be outside the P1 directory.")
    if bundle_dir.exists():
        raise FileExistsError(f"Bundle directory already exists; choose a new path: {bundle_dir}")

    dataset_id = f"{username}/{dataset_slug}"
    kernel_id = f"{username}/{kernel_slug}"
    upload_root = bundle_dir / "dataset"
    kernel_root = bundle_dir / "kernel"
    staged_p1 = upload_root / "AI_Based_Oral_Absorption_Prediction_System"
    processed = staged_p1 / "data" / "processed"
    staged_p2 = staged_p1 / "p2_chemberta"
    processed.mkdir(parents=True)
    staged_p2.mkdir(parents=True)
    kernel_root.mkdir(parents=True)

    for filename in P1_DATA_FILES:
        source = p1_dir / "data" / "processed" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required P1 file missing: {source}")
        shutil.copy2(source, processed / filename)
    shutil.copy2(p1_dir / "masked_weighted_loss.py", staged_p1 / "masked_weighted_loss.py")
    for filename in P2_FILES:
        shutil.copy2(P2_ROOT / filename, staged_p2 / filename)
    shutil.copy2(P2_ROOT / "kaggle_train.py", kernel_root / "kaggle_train.py")

    dataset_metadata = {
        "title": "AI-Based Oral Absorption Prediction System",
        "id": dataset_id,
        "licenses": [{"name": "unknown"}],
        "description": (
            "Private AI-Based Oral Absorption Prediction System training bundle containing copies of the approved P1 cleaned dataset, "
            "frozen split manifest, training-only class weights, and P2 ChemBERTa scripts. "
            "Not intended for public sharing."
        ),
    }
    (upload_root / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    kernel_metadata = {
        "id": kernel_id,
        "title": "AI-Based Oral Absorption Prediction System",
        "code_file": "kaggle_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_root / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2), encoding="utf-8")
    print(f"Prepared private Kaggle dataset files: {upload_root}")
    print(f"Prepared private Kaggle kernel files: {kernel_root}")
    print("Next: kaggle datasets create -p <bundle>/dataset -r zip -t")
    print("Then upload without running: kaggle kernels push -p <bundle>/kernel --no-run")
    print("When ready to start training: kaggle kernels push -p <bundle>/kernel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage P1 copies and the P2 trainer for private Kaggle use.")
    parser.add_argument("--p1-dir", type=Path, default=P1_ROOT)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--username", required=True, help="Kaggle account username slug (not a credential).")
    parser.add_argument("--dataset-slug", default="ai-based-oral-absorption-data")
    parser.add_argument("--kernel-slug", default="ai-based-oral-absorption-prediction-system")
    args = parser.parse_args()
    build_bundle(args.p1_dir, args.bundle_dir, args.username, args.dataset_slug, args.kernel_slug)


if __name__ == "__main__":
    main()
