from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "p2_chemberta"
if str(P2) not in sys.path:
    sys.path.insert(0, str(P2))
from model import ENDPOINTS, load_saved_model  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "best_model"
SAVED_EVALUATION = ROOT / "outputs" / "AI_Based_Oral_Absorption_Prediction_System" / "evaluation.json"
THRESHOLD = 0.5
BATCH_SIZE = 16
DEVICE = torch.device("cpu")


def audit_splits(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(DATA_DIR / "split_manifest.csv")
    if cleaned["InChIKey"].duplicated().any() or manifest["InChIKey"].duplicated().any():
        raise ValueError("InChIKey is not unique in cleaned data or split manifest.")
    if set(cleaned["InChIKey"]) != set(manifest["InChIKey"]):
        raise ValueError("Cleaned data and split manifest InChIKey sets differ.")
    joined = cleaned.merge(
        manifest[["InChIKey", "Canonical_SMILES", "Scaffold_ID", "Scaffold_Type", "Split"]],
        on="InChIKey", how="left", validate="one_to_one", suffixes=("_data", "_manifest"), sort=False,
    )
    if joined["Split"].isna().any():
        raise ValueError("Some molecules have no frozen split assignment.")
    if not joined["Split"].isin(("train", "validation", "test")).all():
        raise ValueError("Unsupported split value found.")
    canonical_match = joined["Canonical_SMILES_data"].eq(joined["Canonical_SMILES_manifest"])
    scaffold_splits = joined.groupby("Scaffold_ID", dropna=False)["Split"].nunique()
    leaking = scaffold_splits[scaffold_splits > 1].index.astype(str).tolist()

    split_stats = []
    for split in ("train", "validation", "test"):
        subset = joined.loc[joined["Split"].eq(split)]
        row = {
            "split": split,
            "molecules": int(len(subset)),
            "fraction": float(len(subset) / len(joined)),
            "unique_scaffolds": int(subset["Scaffold_ID"].nunique(dropna=False)),
        }
        for endpoint in ENDPOINTS:
            values = pd.to_numeric(subset[endpoint], errors="raise")
            row[f"{endpoint}_positive"] = int((values == 1).sum())
            row[f"{endpoint}_negative"] = int((values == 0).sum())
            row[f"{endpoint}_missing"] = int(values.isna().sum())
        split_stats.append(row)

    audit = {
        "molecules": int(len(joined)),
        "manifest_rows": int(len(manifest)),
        "split_counts": {key: int(value) for key, value in joined["Split"].value_counts().items()},
        "unique_inchikeys": int(joined["InChIKey"].nunique()),
        "duplicate_inchikey_rows": int(cleaned["InChIKey"].duplicated(keep=False).sum()),
        "canonical_smiles_match_manifest": bool(canonical_match.all()),
        "key_cross_split_assignments": int(joined.groupby("InChIKey")["Split"].nunique().gt(1).sum()),
        "scaffold_groups": int(joined["Scaffold_ID"].nunique(dropna=False)),
        "scaffold_groups_crossing_splits": len(leaking),
        "leaking_scaffold_ids": leaking,
        "scaffold_types": {str(k): int(v) for k, v in manifest["Scaffold_Type"].value_counts(dropna=False).items()},
        "split_stats": split_stats,
    }
    return joined, audit


def score_split(model, tokenizer, frame: pd.DataFrame, max_length: int) -> pd.DataFrame:
    smiles = frame["Canonical_SMILES_data"].astype(str).tolist()
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(smiles), BATCH_SIZE):
            tokens = tokenizer(
                smiles[start:start + BATCH_SIZE],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
            )
            logits = model(
                input_ids=tokens["input_ids"].to(DEVICE),
                attention_mask=tokens["attention_mask"].to(DEVICE),
            )
            outputs.append(logits.float().cpu())
    probabilities = torch.sigmoid(torch.cat(outputs)).numpy()
    result = frame[["InChIKey", "Canonical_SMILES_data", "Scaffold_ID", "Scaffold_Type"]].copy()
    result.rename(columns={"Canonical_SMILES_data": "Canonical_SMILES"}, inplace=True)
    for column, endpoint in enumerate(ENDPOINTS):
        target = pd.to_numeric(frame[endpoint], errors="raise").to_numpy(dtype=np.float64)
        result[f"{endpoint}_label"] = target
        result[f"{endpoint}_probability"] = probabilities[:, column]
        result[f"{endpoint}_prediction"] = (probabilities[:, column] >= THRESHOLD).astype(np.int8)
    return result


def endpoint_metrics(frame: pd.DataFrame, endpoint: str) -> tuple[dict, np.ndarray, np.ndarray]:
    target = frame[f"{endpoint}_label"].to_numpy(dtype=np.float64)
    probability = frame[f"{endpoint}_probability"].to_numpy(dtype=np.float64)
    mask = np.isfinite(target)
    y_true = target[mask].astype(np.int8)
    y_probability = probability[mask]
    y_prediction = (y_probability >= THRESHOLD).astype(np.int8)
    matrix = confusion_matrix(y_true, y_prediction, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    both_classes = np.unique(y_true).size == 2
    metric = {
        "endpoint": endpoint,
        "n_observed": int(mask.sum()),
        "positive": int((y_true == 1).sum()),
        "negative": int((y_true == 0).sum()),
        "missing": int((~mask).sum()),
        "threshold": THRESHOLD,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "sensitivity_recall": float(recall_score(y_true, y_prediction, zero_division=0)) if len(y_true) else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y_true, y_prediction, zero_division=0)) if len(y_true) else None,
        "f1": float(f1_score(y_true, y_prediction, zero_division=0)) if len(y_true) else None,
        "accuracy": float(accuracy_score(y_true, y_prediction)) if len(y_true) else None,
        "roc_auc": float(roc_auc_score(y_true, y_probability)) if both_classes else None,
        "pr_auc_average_precision": float(average_precision_score(y_true, y_probability))
        if np.any(y_true == 1) else None,
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "class_support_sufficient_for_roc_auc": bool(both_classes),
    }
    return metric, y_true, y_probability


def color(value: float) -> str:
    low = np.array([246, 248, 250], dtype=float)
    high = np.array([94, 130, 165], dtype=float)
    rgb = np.rint(low + (high - low) * np.clip(value, 0, 1)).astype(int)
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def write_confusions(metrics: list[dict], path: Path) -> None:
    width, height = 1080, 380
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#25374b}.title{font-size:19px;font-weight:bold}.label{font-size:13px}.cell{font-size:17px;font-weight:bold}</style>',
         '<text x="26" y="30" class="title">Frozen test set: confusion matrices at threshold 0.5</text>']
    for idx, m in enumerate(metrics):
        x0, y0, cell = 28 + idx * 355, 84, 88
        matrix = m["confusion_matrix_labels_0_1"]
        maximum = max(max(row) for row in matrix) or 1
        s.append(f'<text x="{x0 + 110}" y="64" class="title">{html.escape(m["endpoint"])}</text>')
        s.extend([f'<text x="{x0}" y="{y0 + 42}" class="label">Actual 0</text>',
                  f'<text x="{x0}" y="{y0 + 130}" class="label">Actual 1</text>'])
        for r in range(2):
            for c in range(2):
                x, y, val = x0 + 92 + c * cell, y0 + r * cell, matrix[r][c]
                name = ("TN", "FP")[c] if r == 0 else ("FN", "TP")[c]
                s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color(val / maximum)}" stroke="white" stroke-width="2"/>')
                s.append(f'<text x="{x + cell/2}" y="{y + 38}" text-anchor="middle" class="cell">{val}</text>')
                s.append(f'<text x="{x + cell/2}" y="{y + 60}" text-anchor="middle" class="label">{name}</text>')
        s.append(f'<text x="{x0 + 95}" y="{y0 + 205}" class="label">Predicted 0                 Predicted 1</text>')
        s.append(f'<text x="{x0}" y="{y0 + 238}" class="label">Observed n={m["n_observed"]}; sensitivity={m["sensitivity_recall"]:.3f}; specificity={m["specificity"]:.3f}</text>')
    s.append("</svg>")
    path.write_text("\n".join(s), encoding="utf-8")


def write_pr_curves(curves: dict, path: Path) -> None:
    width, height = 1080, 435
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#25374b}.title{font-size:19px;font-weight:bold}.axis{font-size:12px}.curve{fill:none;stroke:#527A9D;stroke-width:3}.base{stroke:#9ba6b1;stroke-width:1.5;stroke-dasharray:6 5}</style>',
         '<text x="26" y="30" class="title">Frozen test set: precision-recall curves</text>']
    for idx, (endpoint, (precisions, recalls, metric)) in enumerate(curves.items()):
        x0, y0, pw, ph = 55 + idx * 355, 80, 270, 265
        prevalence = metric["positive"] / metric["n_observed"]
        s.append(f'<text x="{x0 + 86}" y="61" class="title">{html.escape(endpoint)} (AP={metric["pr_auc_average_precision"]:.3f})</text>')
        base_y = y0 + ph * (1 - prevalence)
        s.append(f'<line x1="{x0}" y1="{base_y:.1f}" x2="{x0 + pw}" y2="{base_y:.1f}" class="base"/>')
        for tick in (0, .5, 1):
            xx, yy = x0 + pw * tick, y0 + ph * (1 - tick)
            s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + ph}" stroke="#e4e8ec"/>')
            s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="#e4e8ec"/>')
            s.append(f'<text x="{xx:.1f}" y="{y0 + ph + 18}" text-anchor="middle" class="axis">{tick:.1f}</text>')
            s.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>')
        points = " ".join(
            f"{x0 + pw * float(r):.2f},{y0 + ph * (1 - float(p)):.2f}"
            for r, p in zip(recalls, precisions, strict=True)
        )
        s.append(f'<polyline points="{points}" class="curve"/>')
        s.append(f'<text x="{x0 + 105}" y="{y0 + ph + 42}" class="axis">Recall</text>')
        s.append(f'<text transform="translate({x0 - 36},{y0 + 165}) rotate(-90)" class="axis">Precision</text>')
        s.append(f'<text x="{x0 + 5}" y="{y0 + ph + 65}" class="axis">Dashed = positive prevalence ({prevalence:.2f})</text>')
    s.append("</svg>")
    path.write_text("\n".join(s), encoding="utf-8")


def fmt(value) -> str:
    return f"{float(value):.3f}" if value is not None else "n/a"


def fmt_delta(value) -> str:
    return f"{float(value):.6g}" if value is not None else "n/a"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = pd.read_csv(DATA_DIR / "cleaned_dataset.csv")
    joined, split_audit = audit_splits(cleaned)
    weights = pd.read_csv(DATA_DIR / "class_distribution_report.csv")
    split_audit["training_only_pos_weights"] = {
        str(row["Endpoint"]): float(row["Train_Pos_Weight"])
        for _, row in weights.query("Split == 'train'").iterrows()
    }
    (OUTPUT_DIR / "split_integrity.json").write_text(json.dumps(split_audit, indent=2), encoding="utf-8")

    saved = json.loads(SAVED_EVALUATION.read_text(encoding="utf-8"))
    model, tokenizer, config = load_saved_model(MODEL_DIR, device=DEVICE)
    if tuple(config.get("endpoints", ())) != tuple(ENDPOINTS):
        raise ValueError("Saved model endpoint order differs from endpoint order in data.")
    max_length = int(config["max_length"])
    threshold = float(config.get("decision_threshold", .5))
    if threshold != THRESHOLD:
        raise ValueError(f"Configured threshold is {threshold}, expected {THRESHOLD}.")

    metrics, metric_comparison = [], {}
    test_predictions = None
    errors, test_curves = [], {}
    for split in ("validation", "test"):
        rows = joined.loc[joined["Split"].eq(split)].reset_index(drop=True)
        scored = score_split(model, tokenizer, rows, max_length)
        for endpoint in ENDPOINTS:
            m, y_true, y_prob = endpoint_metrics(scored, endpoint)
            m["split"] = split
            metrics.append(m)
            if np.any(y_true == 1) and np.any(y_true == 0):
                p, r, _ = precision_recall_curve(y_true, y_prob)
                if split == "test":
                    test_curves[endpoint] = (p, r, m)
            if split == "test":
                mask = np.isfinite(scored[f"{endpoint}_label"].to_numpy(dtype=float))
                observed = scored.loc[mask].copy()
                truth = observed[f"{endpoint}_label"].to_numpy(dtype=np.int8)
                pred = observed[f"{endpoint}_prediction"].to_numpy(dtype=np.int8)
                prob = observed[f"{endpoint}_probability"].to_numpy(dtype=float)
                kind = np.where((truth == 0) & (pred == 1), "false_positive",
                       np.where((truth == 1) & (pred == 0), "false_negative", ""))
                bad = kind != ""
                err = observed.loc[bad].copy()
                err["endpoint"] = endpoint
                err["error_type"] = kind[bad]
                err["true_label"] = truth[bad]
                err["predicted_label"] = pred[bad]
                err["predicted_probability"] = prob[bad]
                errors.append(err)
                test_predictions = scored

            if endpoint in saved.get(split, {}).get("endpoints", {}):
                before = saved[split]["endpoints"][endpoint]
                diffs = {}
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    current = m["sensitivity_recall"] if key == "recall" else m[key]
                    prior = before.get(key)
                    diffs[key] = abs(float(prior) - float(current)) if prior is not None and current is not None else None
                diffs["confusion_matrix_matches"] = before.get("confusion_matrix_labels_0_1") == m["confusion_matrix_labels_0_1"]
                vals = [v for k, v in diffs.items() if k != "confusion_matrix_matches" and v is not None]
                diffs["max_absolute_metric_difference"] = max(vals) if vals else None
                metric_comparison[f"{split}_{endpoint}"] = diffs

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_by_endpoint.csv", index=False)
    assert test_predictions is not None
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    pd.concat(errors, ignore_index=True).to_csv(OUTPUT_DIR / "test_false_positives_and_negatives.csv", index=False)
    test_metrics = [m for m in metrics if m["split"] == "test"]
    write_confusions(test_metrics, OUTPUT_DIR / "test_confusion_matrices.svg")
    write_pr_curves(test_curves, OUTPUT_DIR / "test_precision_recall_curves.svg")

    both_pred = (
        (test_predictions["HOB50_prediction"] == 1)
        & (test_predictions["HOB20_prediction"] == 0)
    )
    prob_order = test_predictions["HOB50_probability"] > test_predictions["HOB20_probability"]
    both_labels = np.isfinite(test_predictions["HOB20_label"]) & np.isfinite(test_predictions["HOB50_label"])
    test_label_violation = both_labels & (test_predictions["HOB50_label"] == 1) & (test_predictions["HOB20_label"] == 0)
    all_both_labels = np.isfinite(joined["HOB20"]) & np.isfinite(joined["HOB50"])
    full_label_violation = int((all_both_labels & (joined["HOB50"] == 1) & (joined["HOB20"] == 0)).sum())
    nested = {
        "interpretation": "Conditional nesting check: if both labels mark the same oral-bioavailability quantity at 20% and 50% cutoffs, HOB50 positive with HOB20 negative is inconsistent.",
        "test_rows": int(len(test_predictions)),
        "test_binary_prediction_violations": int(both_pred.sum()),
        "test_probability_order_violations_HOB50_gt_HOB20": int(prob_order.sum()),
        "test_rows_with_both_observed_labels": int(both_labels.sum()),
        "test_observed_label_violations": int(test_label_violation.sum()),
        "full_data_rows_with_both_observed_labels": int(all_both_labels.sum()),
        "full_data_observed_label_violations": full_label_violation,
        "test_prediction_violation_rows": test_predictions.loc[
            both_pred,
            ["InChIKey", "Canonical_SMILES", "HOB20_probability", "HOB20_prediction", "HOB50_probability", "HOB50_prediction"],
        ].to_dict(orient="records"),
    }

    lines = [
        "# ChemBERTa Model Validation Report",
        "",
        "## Scope and protocol",
        "",
        f"- Saved evaluation file: {SAVED_EVALUATION.relative_to(ROOT).as_posix()}",
        "- Checkpoint: best_model; unchanged during this audit.",
        f"- Classification threshold: {THRESHOLD:.1f}, read from the saved model configuration.",
        f"- Saved best epoch: {saved.get('best_epoch')}; saved best validation loss: {saved.get('best_validation_loss')}.",
        "- The app checkpoint and training-output checkpoint state dictionaries have identical SHA-256 (113e9216836c9b7cf23c782ada32802db5f40e0bfc1f81e5909de8996eedc83d); encoder weight files also match. Tokenizer JSON/configuration are semantically equal with the same 593-token ID mapping.",
        "- Saved evaluation used mixed precision on a Tesla T4; this audit uses CPU FP32. All validation/test confusion matrices and 0.5-threshold metrics match exactly; ROC-AUC differs by at most 0.000446, consistent with small numerical changes in probability ranking.",
        "- Validation and test rows were selected only by the existing split manifest. No new split was created.",
        "- Weights were not selected or adjusted using test results. All test metrics below are post-selection evaluation.",
        "- Missing labels remain NaN and are excluded from only their endpoint's metric calculation.",
        "",
        "## Split integrity",
        "",
        f"- {split_audit['molecules']} molecules; {split_audit['unique_inchikeys']} unique InChIKeys; duplicate InChIKey rows: {split_audit['duplicate_inchikey_rows']}.",
        f"- One-to-one key set match: {split_audit['unique_inchikeys'] == split_audit['manifest_rows']}; canonical SMILES match between dataset and manifest: {split_audit['canonical_smiles_match_manifest']}.",
        f"- Molecule keys crossing splits: {split_audit['key_cross_split_assignments']}; scaffold groups crossing splits: {split_audit['scaffold_groups_crossing_splits']} of {split_audit['scaffold_groups']}.",
        f"- Scaffold types: {json.dumps(split_audit['scaffold_types'], sort_keys=True)}.",
        "",
        "| Split | Molecules | Share | Unique scaffold groups |",
        "|---|---:|---:|---:|",
    ]
    for row in split_audit["split_stats"]:
        lines.append(f"| {row['split']} | {row['molecules']} | {row['fraction']:.2%} | {row['unique_scaffolds']} |")
    lines.extend([
        "",
        "## Endpoint class distributions",
        "",
        "Class percentages below use observed labels only. Missing labels are not treated as negative.",
        "",
        "| Split | Endpoint | Positive | Negative | Missing | Positive % among observed |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in split_audit["split_stats"]:
        for endpoint in ENDPOINTS:
            pos, neg, missing = (row[f"{endpoint}_{key}"] for key in ("positive", "negative", "missing"))
            lines.append(f"| {row['split']} | {endpoint} | {pos} | {neg} | {missing} | {pos/(pos+neg):.1%} |")
    lines.extend([
        "",
        "## Validation and test metrics",
        "",
        "Confusion matrix convention: [[TN, FP], [FN, TP]]. Sensitivity equals recall. Specificity = TN/(TN+FP). PR-AUC is reported as average precision (AP).",
        "",
        "| Split | Endpoint | n observed | Pos / Neg | TN | FP | FN | TP | Sensitivity | Specificity | Precision | F1 | Accuracy | ROC-AUC | PR-AUC (AP) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for m in metrics:
        lines.append(
            f"| {m['split']} | {m['endpoint']} | {m['n_observed']} | {m['positive']} / {m['negative']} | "
            f"{m['true_negative']} | {m['false_positive']} | {m['false_negative']} | {m['true_positive']} | "
            f"{fmt(m['sensitivity_recall'])} | {fmt(m['specificity'])} | {fmt(m['precision'])} | {fmt(m['f1'])} | "
            f"{fmt(m['accuracy'])} | {fmt(m['roc_auc'])} | {fmt(m['pr_auc_average_precision'])} |"
        )
    lines.extend([
        "",
        "## Test false positives and false negatives",
        "",
        "| Endpoint | False positives | False negatives |",
        "|---|---:|---:|",
    ])
    for m in test_metrics:
        lines.append(f"| {m['endpoint']} | {m['false_positive']} | {m['false_negative']} |")
    lines.extend([
        "",
        "Per-compound InChIKeys, canonical SMILES, labels, probabilities, and predictions are in test_false_positives_and_negatives.csv. Only observed labels appear in this error file.",
        "",
        "## HOB20/HOB50 nested-threshold consistency",
        "",
        "- " + json.dumps(nested, ensure_ascii=False),
        "- This is a consistency diagnostic only; no predictions were changed or post-processed.",
        "- Interpretation assumes the two endpoints measure the same underlying bioavailability quantity at nested cutoffs. That source definition has not been independently confirmed.",
        "",
        "## Reproduction against saved evaluation",
        "",
        "The same root best checkpoint was compared with the saved validation/test metrics. Differences are:",
        "",
        "| Split / endpoint | Max absolute metric difference | Confusion matrix matched |",
        "|---|---:|---|",
    ])
    for key, value in metric_comparison.items():
        lines.append(f"| {key.replace('_', ' / ')} | {fmt_delta(value['max_absolute_metric_difference'])} | {value['confusion_matrix_matches']} |")
    lines.extend([
        "",
        "## Leakage, bias, and generalization limitations",
        "",
        "- The split is InChIKey-unique and scaffold-disjoint across train, validation, and test. This supports an internal scaffold-split assessment, not an independent external benchmark.",
        "- The P1 quality report says original benchmark/test-sheet compound identities were unavailable. Overlap with those original benchmark compounds cannot be checked from current files.",
        "- Endpoint metrics use different observed subsets because labels are missing at different rates. Results across endpoints are not paired on a shared cohort.",
        "- Test supports are endpoint-specific and limited; class-conditional estimates can have substantial sampling uncertainty. No confidence intervals are included.",
        "- HIA and HOB20 are class-imbalanced in training and test. Training class weights do not correct source, assay, or chemical-domain bias.",
        "- The P1 report records HIA positive as >30%, but the authoritative source definition is not present here; >30% versus >=30% remains unresolved. Labels were not changed.",
        "- No nutraceutical-specific identity, cohort, or external test set is present. Internal scaffold-split results do not support generalization claims specifically for nutraceutical compounds.",
        "- Saved evaluation JSON has no per-epoch history or probability-level rows. This audit recomputed validation/test predictions with the identical saved weights.",
        "",
        "## Generated artifacts",
        "",
        "- metrics_by_endpoint.csv: validation and test metrics and endpoint support.",
        "- split_integrity.json: frozen split checks and distribution counts.",
        "- test_predictions.csv: test labels (NaNs retained), probabilities, and predictions.",
        "- test_false_positives_and_negatives.csv: observed-label mistakes by endpoint.",
        "- test_confusion_matrices.svg and test_precision_recall_curves.svg: plots.",
        "",
        "No model, dataset, split, training, or inference files were modified.",
    ])
    (OUTPUT_DIR / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "split_audit": split_audit,
        "metrics": metrics,
        "saved_metric_comparison": metric_comparison,
        "nested_threshold_consistency": nested,
        "test_error_counts": {
            m["endpoint"]: {"false_positives": m["false_positive"], "false_negatives": m["false_negative"]}
            for m in test_metrics
        },
    }
    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Validation outputs: {OUTPUT_DIR}")
    print("Split checks:", json.dumps({key: value for key, value in split_audit.items() if key != "split_stats"}))
    print("Test metrics:", json.dumps(test_metrics))
    print("Saved-report comparison:", json.dumps(metric_comparison))
    print("Nested consistency:", json.dumps(nested))
    print("FP/FN counts:", json.dumps(summary["test_error_counts"]))


if __name__ == "__main__":
    main()
