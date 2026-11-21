"""Validate and clean the AI-Based Oral Absorption Prediction System multitask model table.

Run from the project root with ``python data_preprocessing.py``. The original
CSV is read only. Cleaned data and audit files are written to data/processed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "oralabs_model_training.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
ENDPOINTS = ("HIA", "HOB20", "HOB50")
REQUIRED_COLUMNS = ("InChIKey", "SMILES", *ENDPOINTS)


def _parse_label(value: Any, endpoint: str, source_row: int) -> float:
    """Return a validated 0/1 label or NaN for a genuinely missing value."""
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Ambiguous {endpoint} label at CSV row {source_row}: {value!r}. "
            "Only 0, 1, or blank is accepted."
        ) from exc
    if numeric not in (0.0, 1.0):
        raise ValueError(
            f"Ambiguous {endpoint} label at CSV row {source_row}: {value!r}. "
            "Only 0, 1, or blank is accepted."
        )
    return numeric


def _structure_record(smiles_value: Any) -> tuple[Chem.Mol, str, str] | None:
    """Parse and sanitize a SMILES; return molecule, canonical SMILES, InChIKey."""
    if pd.isna(smiles_value) or not str(smiles_value).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles_value).strip(), sanitize=True)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not canonical or not inchi_key:
        return None
    return mol, canonical, inchi_key


def clean_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate structures/labels, deduplicate by computed InChIKey, and audit."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Model dataset not found: {input_path}")
    output_names = (
        "cleaned_dataset.csv", "invalid_records.csv", "duplicate_audit.csv",
        "label_conflict_audit.csv",
    )
    resolved_input = input_path.resolve()
    collisions = [name for name in output_names if (output_dir / name).resolve() == resolved_input]
    if collisions:
        raise ValueError(
            f"Input dataset would be overwritten by preprocessing output: {collisions}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path, dtype={"InChIKey": "string", "SMILES": "string"})
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    logging.info("Input: %s", input_path)
    logging.info("Original dimensions: %d rows x %d columns", *raw.shape)
    logging.info("Columns and inferred types:\n%s", raw.dtypes.to_string())
    missing_labels_in_source = {
        endpoint: int(raw[endpoint].isna().sum() + (raw[endpoint].astype("string").str.strip() == "").sum())
        for endpoint in ENDPOINTS
    }

    input_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    for index, row in raw.iterrows():
        source_row = int(index) + 2  # CSV header is row 1.
        labels = {endpoint: _parse_label(row[endpoint], endpoint, source_row) for endpoint in ENDPOINTS}
        original_smiles = "" if pd.isna(row["SMILES"]) else str(row["SMILES"]).strip()
        parsed = _structure_record(original_smiles)
        if parsed is None:
            reason = "missing SMILES" if not original_smiles else "RDKit could not parse/sanitize SMILES"
            invalid_records.append({
                "Source_Row": source_row,
                "Input_InChIKey": "" if pd.isna(row["InChIKey"]) else str(row["InChIKey"]),
                "SMILES": original_smiles,
                **labels,
                "Exclusion_Reason": reason,
            })
            continue
        _, canonical_smiles, computed_key = parsed
        input_key = "" if pd.isna(row["InChIKey"]) else str(row["InChIKey"]).strip()
        input_records.append({
            "Source_Row": source_row,
            "Input_InChIKey": input_key,
            "InChIKey": computed_key,
            "InChIKey_Matches_Input": bool(input_key and input_key == computed_key),
            "SMILES": original_smiles,
            "Canonical_SMILES": canonical_smiles,
            **labels,
        })

    valid_frame = pd.DataFrame(input_records)
    duplicate_audit: list[dict[str, Any]] = []
    conflict_audit: list[dict[str, Any]] = []
    cleaned: list[dict[str, Any]] = []
    if not valid_frame.empty:
        for inchi_key, group in valid_frame.groupby("InChIKey", sort=True, dropna=False):
            record = group.iloc[0].to_dict()
            record["Source_Rows"] = ";".join(str(int(value)) for value in group["Source_Row"].tolist())
            record["Input_InChIKeys"] = ";".join(sorted({str(v) for v in group["Input_InChIKey"] if str(v)}))
            record["InChIKey_Matches_Input"] = bool(group["InChIKey_Matches_Input"].all())
            conflicts: list[str] = []
            for endpoint in ENDPOINTS:
                observed = sorted({float(v) for v in group[endpoint].dropna().tolist()})
                if len(observed) > 1:
                    record[endpoint] = float("nan")
                    conflicts.append(endpoint)
                    conflict_audit.append({
                        "InChIKey": inchi_key,
                        "Endpoint": endpoint,
                        "Conflicting_Labels": ";".join(str(int(value)) for value in observed),
                        "Source_Rows": record["Source_Rows"],
                        "Resolution": "Set only this endpoint to missing; retain other consistent labels.",
                    })
                elif observed:
                    record[endpoint] = observed[0]
                else:
                    record[endpoint] = float("nan")
            if len(group) > 1:
                duplicate_audit.append({
                    "InChIKey": inchi_key,
                    "Canonical_SMILES": record["Canonical_SMILES"],
                    "Duplicate_Record_Count": int(len(group)),
                    "Conflicting_Endpoints": ";".join(conflicts),
                    "Source_Rows": record["Source_Rows"],
                    "Resolution": (
                        "Coalesced identical labels; conflicting endpoint labels set missing."
                        if conflicts else "Coalesced duplicate records with identical labels."
                    ),
                })
            cleaned.append(record)

    cleaned_frame = pd.DataFrame(cleaned)
    output_columns = [
        "Source_Row", "Source_Rows", "Input_InChIKey", "Input_InChIKeys",
        "InChIKey", "InChIKey_Matches_Input", "SMILES", "Canonical_SMILES", *ENDPOINTS,
    ]
    cleaned_frame = cleaned_frame.reindex(columns=output_columns)
    cleaned_path = output_dir / "cleaned_dataset.csv"
    invalid_path = output_dir / "invalid_records.csv"
    duplicate_path = output_dir / "duplicate_audit.csv"
    conflict_path = output_dir / "label_conflict_audit.csv"
    cleaned_frame.to_csv(cleaned_path, index=False)
    pd.DataFrame(invalid_records, columns=[
        "Source_Row", "Input_InChIKey", "SMILES", *ENDPOINTS, "Exclusion_Reason"
    ]).to_csv(invalid_path, index=False)
    pd.DataFrame(duplicate_audit, columns=[
        "InChIKey", "Canonical_SMILES", "Duplicate_Record_Count",
        "Conflicting_Endpoints", "Source_Rows", "Resolution"
    ]).to_csv(duplicate_path, index=False)
    pd.DataFrame(conflict_audit, columns=[
        "InChIKey", "Endpoint", "Conflicting_Labels", "Source_Rows", "Resolution"
    ]).to_csv(conflict_path, index=False)

    stats = {
        "source_path": str(input_path),
        "source_rows": int(len(raw)),
        "source_columns": list(raw.columns),
        "missing_labels_in_source": missing_labels_in_source,
        "valid_rows_before_dedup": int(len(valid_frame)),
        "invalid_rows": int(len(invalid_records)),
        "unique_molecules": int(len(cleaned_frame)),
        "duplicate_groups": int(len(duplicate_audit)),
        "duplicate_extra_rows": int(len(valid_frame) - len(cleaned_frame)),
        "conflicting_duplicate_endpoints": int(len(conflict_audit)),
        "key_mismatches": int((~valid_frame["InChIKey_Matches_Input"]).sum()) if len(valid_frame) else 0,
        "valid_smiles": int(len(valid_frame)),
        "invalid_path": str(invalid_path),
        "duplicate_audit_path": str(duplicate_path),
        "conflict_audit_path": str(conflict_path),
        "cleaned_path": str(cleaned_path),
        "label_semantics": {
            "HIA": "The P1 labels agree with the supplied SI2_HIA_new_unique_682.csv source values for 663 matched source records: class 1 is >30%, class 0 is <=30%, with all 3 exact-30% records labeled 0. The companion OralAbsPredict app documentation instead states class 1 is >=30% and class 0 is <30%; preserve this source/documentation boundary conflict for resolution before modeling.",
            "HOB20": "P1 assumes class 1 means HOB >= 20% and class 0 means < 20%, consistent with the companion OralAbsPredict app documentation; row-level source values and label mapping have not been independently verified in this table.",
            "HOB50": "P1 assumes class 1 means HOB >= 50% and class 0 means < 50%, consistent with the companion OralAbsPredict app documentation; row-level source values and label mapping have not been independently verified in this table.",
        },
    }
    stats["missing_labels_after_cleaning"] = {
        endpoint: int(cleaned_frame[endpoint].isna().sum()) for endpoint in ENDPOINTS
    }
    logging.info(
        "Structures: valid=%d; invalid=%d; duplicate groups=%d; extra duplicate rows=%d; label conflicts=%d",
        stats["valid_smiles"], stats["invalid_rows"], stats["duplicate_groups"],
        stats["duplicate_extra_rows"], stats["conflicting_duplicate_endpoints"],
    )
    logging.info("Saved cleaned data: %s (%d molecules)", cleaned_path, len(cleaned_frame))
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_dataset(args.input.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
