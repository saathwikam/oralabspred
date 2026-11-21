"""Validation-only threshold feasibility analysis for HOB20 and HOB50."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_validation as baseline  # noqa: E402


def metrics_at(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive": int((y_true == 1).sum()),
        "negative": int((y_true == 0).sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probability)),
    }


def main() -> None:
    out = HERE
    data = pd.read_csv(ROOT / "data" / "processed" / "cleaned_dataset.csv")
    joined, split_audit = baseline.audit_splits(data)
    validation = joined.loc[joined["Split"].eq("validation")].reset_index(drop=True)
    model, tokenizer, config = baseline.load_saved_model(baseline.MODEL_DIR, baseline.DEVICE)
    scored = baseline.score_split(model, tokenizer, validation, int(config["max_length"]))
    # Retain frozen labels, including NaNs, and all validation probabilities.
    for endpoint in ("HOB20", "HOB50"):
        scored[f"{endpoint}_label"] = pd.to_numeric(validation[endpoint], errors="raise").to_numpy()
    scored.to_csv(out / "hob_validation_predictions.csv", index=False)

    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    grid = sorted(set(np.round(np.arange(0.10, 0.901, 0.05), 2).tolist() + [0.5]))
    for endpoint in ("HOB20", "HOB50"):
        labels = scored[f"{endpoint}_label"].to_numpy(dtype=float)
        probs = scored[f"{endpoint}_probability"].to_numpy(dtype=float)
        observed = np.isfinite(labels)
        y, p = labels[observed].astype(np.int8), probs[observed]
        for threshold in grid:
            rows.append({"endpoint": endpoint, **metrics_at(y, p, threshold)})
        # Include probability-derived decision boundaries for actual empirical optima.
        candidate_thresholds = np.unique(np.r_[0.0, p, np.nextafter(1.0, 2.0)])
        candidates = [metrics_at(y, p, float(t)) for t in candidate_thresholds]
        summaries[endpoint] = {
            "n_validation": int(len(validation)),
            "n_observed": int(len(y)),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "missing": int((~observed).sum()),
            "positive_rate_observed": float((y == 1).mean()),
            "at_0_5": metrics_at(y, p, 0.5),
            "max_f1_validation_candidate": max(candidates, key=lambda x: (x["f1"], x["threshold"])),
            "max_balanced_accuracy_validation_candidate": max(
                candidates, key=lambda x: (x["balanced_accuracy"], x["threshold"])
            ),
            "max_f1_on_grid": max((r for r in rows if r["endpoint"] == endpoint), key=lambda x: x["f1"]),
        }

    both = np.isfinite(scored["HOB20_label"]) & np.isfinite(scored["HOB50_label"])
    both_count = int(both.sum())
    label_violation = (scored.loc[both, "HOB50_label"] == 1) & (scored.loc[both, "HOB20_label"] == 0)
    pred20 = scored["HOB20_probability"].to_numpy() >= 0.5
    pred50 = scored["HOB50_probability"].to_numpy() >= 0.5
    summaries["nested_and_coverage"] = {
        "validation_total": int(len(scored)),
        "HOB20_observed": int(np.isfinite(scored["HOB20_label"]).sum()),
        "HOB50_observed": int(np.isfinite(scored["HOB50_label"]).sum()),
        "both_observed": both_count,
        "neither_observed": int((~np.isfinite(scored["HOB20_label"]) & ~np.isfinite(scored["HOB50_label"])).sum()),
        "HOB20_only": int((np.isfinite(scored["HOB20_label"]) & ~np.isfinite(scored["HOB50_label"])).sum()),
        "HOB50_only": int((~np.isfinite(scored["HOB20_label"]) & np.isfinite(scored["HOB50_label"])).sum()),
        "observed_nested_label_violations_HOB50_pos_HOB20_neg": int(label_violation.sum()),
        "both_label_observation_patterns": {
            f"HOB20={a},HOB50={b}": int(((scored.loc[both, "HOB20_label"] == a) & (scored.loc[both, "HOB50_label"] == b)).sum())
            for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))
        },
        "independent_head_binary_prediction_violations_at_0_5": int((pred50 & ~pred20).sum()),
        "probability_order_violations_HOB50_gt_HOB20": int((scored["HOB50_probability"] > scored["HOB20_probability"]).sum()),
        "probability_order_violations_fraction": float((scored["HOB50_probability"] > scored["HOB20_probability"]).mean()),
    }
    pd.DataFrame(rows).to_csv(out / "hob_validation_threshold_metrics.csv", index=False)
    (out / "hob_validation_feasibility.json").write_text(
        json.dumps({"scope": "validation only", "split_counts": split_audit["split_counts"], "endpoints": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
