from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from data import fit_transform_descriptors, load_v2_data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def prepare(p1_dir: Path = ROOT, output_dir: Path = HERE / "run_artifacts") -> dict:
    p1_dir, output_dir = p1_dir.resolve(), output_dir.resolve()
    frame, _, descriptors, failures = load_v2_data(p1_dir)
    transformed, preprocessor = fit_transform_descriptors(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(output_dir / "descriptor_values.csv", index=False)
    failures.to_csv(output_dir / "descriptor_failures.csv", index=False)
    scaled = pd.DataFrame(transformed, columns=preprocessor.parameters()["descriptor_names"])
    scaled.insert(0, "InChIKey", frame.InChIKey.to_numpy())
    scaled.insert(1, "Split", frame.Split.to_numpy())
    scaled.to_csv(output_dir / "descriptor_values_scaled.csv", index=False)
    prep = preprocessor.parameters()
    prep["fit_dataset_sha256"] = hashlib.sha256(
        (p1_dir / "data" / "processed" / "cleaned_dataset.csv").read_bytes()).hexdigest()
    prep["fit_manifest_sha256"] = hashlib.sha256(
        (p1_dir / "data" / "processed" / "split_manifest.csv").read_bytes()).hexdigest()
    (output_dir / "descriptor_preprocessing.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")
    audit = {"input_rows": int(len(frame)), "descriptor_rows": int(len(descriptors)),
             "failed_calculations": int(len(failures)), "rows_retained": int(len(frame)),
             "failure_policy": "Retain row; missing descriptor vector is median-imputed from training rows only",
             "split_counts": {str(k): int(v) for k, v in frame.Split.value_counts().items()},
             "descriptor_names": prep["descriptor_names"], "preprocessing_fit_split": "train",
             "parameters": prep}
    (output_dir / "descriptor_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
