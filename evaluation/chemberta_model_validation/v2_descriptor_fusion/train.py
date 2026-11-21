from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from data import ENDPOINTS, SPLITS, fit_transform_descriptors, load_v2_data, tokenize_with_descriptors
from model import ChemBERTaDescriptorFusionClassifier
from descriptors import DESCRIPTOR_NAMES
from masked_weighted_loss import MaskedWeightedBCELoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train V2 ChemBERTa + RDKit descriptor late fusion.")
    parser.add_argument("--p1-dir", type=Path, default=ROOT)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "best_model")
    parser.add_argument("--output-dir", type=Path, default=HERE / "run_artifacts")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--descriptor-hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device, amp):
    model.eval()
    logits_all, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            desc, y = batch["descriptors"].to(device), batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(ids, mask, desc)
            logits = logits.float()
            logits_all.append(logits.cpu())
            targets.append(y.cpu().numpy())
    all_logits = torch.cat(logits_all)
    all_targets = torch.as_tensor(np.concatenate(targets), dtype=torch.float32)
    # Match the existing P2 evaluator: calculate masked loss over the full split,
    # not an unweighted mean of minibatch losses.
    loss_value = float(criterion(all_logits, all_targets).item())
    predictions = torch.sigmoid(all_logits).numpy()
    labels = np.concatenate(targets)
    result = {"loss": loss_value, "endpoints": {}, "probabilities": predictions, "labels": labels}
    for column, endpoint in enumerate(ENDPOINTS):
        observed = np.isfinite(labels[:, column])
        y, p = labels[observed, column].astype(np.int8), predictions[observed, column]
        if not len(y):
            result["endpoints"][endpoint] = {"n_observed": 0}
            continue
        predicted = (p >= 0.5).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        both = np.unique(y).size == 2
        result["endpoints"][endpoint] = {
            "n_observed": int(len(y)), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
            "threshold": 0.5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall_sensitivity": float(recall_score(y, predicted, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "pr_auc_average_precision": float(average_precision_score(y, p)) if (y == 1).any() else None,
            "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    return result


def _binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    observed = np.isfinite(labels)
    y, p = labels[observed].astype(np.int8), probability[observed]
    if not len(y):
        return {"n_observed": 0}
    pred = (p >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = np.unique(y).size == 2
    return {"n_observed": int(len(y)), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
            "threshold": 0.5, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "pr_auc_average_precision": float(average_precision_score(y, p)) if (y == 1).any() else None,
            "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def score_baseline_validation(frame: pd.DataFrame, tokenizer, baseline_dir: Path, max_length: int) -> np.ndarray:
    """Score only the validation rows with V1, before training V2; never touches test rows."""
    from p2_chemberta.model import load_saved_model as load_v1
    v1, _, _ = load_v1(baseline_dir, device="cpu")
    v1.eval()
    smiles = frame.Canonical_SMILES.astype(str).tolist()
    all_probs = []
    with torch.inference_mode():
        for start in range(0, len(smiles), 16):
            tokens = tokenizer(smiles[start:start + 16], padding="max_length", truncation=True,
                               max_length=max_length, return_attention_mask=True, return_tensors="pt")
            logits = v1(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
            all_probs.append(torch.sigmoid(logits.float()).numpy())
    return np.concatenate(all_probs)


def env_report(seed, device, amp):
    import rdkit
    free_total = torch.cuda.mem_get_info(device) if device.type == "cuda" else (None, None)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "rdkit": rdkit.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_memory_available_gib_at_start": free_total[0] / 1024**3 if free_total[0] else None,
            "gpu_memory_total_gib": free_total[1] / 1024**3 if free_total[1] else None,
            "cuda_version": torch.version.cuda, "seed": seed, "mixed_precision_enabled": bool(amp)}


def _save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def train(args):
    if min(args.learning_rate, args.batch_size, args.epochs, args.patience,
           args.max_length, args.descriptor_hidden_size) <= 0:
        raise ValueError("Training parameters must be positive")
    p1_dir, baseline_dir, out = args.p1_dir.resolve(), args.baseline_dir.resolve(), args.output_dir.resolve()
    if out == p1_dir or p1_dir in out.parents:
        raise ValueError("V2 outputs must be outside P1; do not overwrite source data")
    baseline_config_path = baseline_dir / "model_config.json"
    if not baseline_config_path.is_file():
        raise FileNotFoundError(f"V1 model configuration missing: {baseline_config_path}")
    baseline_config = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    if int(baseline_config["max_length"]) != args.max_length:
        raise ValueError("V2 max_length must match V1 for a like-for-like comparison")
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.mixed_precision and device.type == "cuda")
    frame, pos_weights, descriptor_table, failures = load_v2_data(p1_dir)
    out.mkdir(parents=True, exist_ok=True)
    descriptor_table.to_csv(out / "descriptor_values.csv", index=False)
    failures.to_csv(out / "descriptor_failures.csv", index=False)
    (out / "descriptor_audit.json").write_text(json.dumps({
        "input_rows": int(len(frame)), "descriptor_rows": int(len(descriptor_table)),
        "failed_calculations": int(len(failures)), "rows_retained": int(len(frame)),
        "handling": "Failed descriptor vectors retained as NaN and median-imputed from train only",
        "descriptor_names": list(DESCRIPTOR_NAMES)}, indent=2), encoding="utf-8")
    transformed, preprocessor = fit_transform_descriptors(frame)
    _save_json(out / "descriptor_preprocessing.json", preprocessor.parameters())
    tokenizer = AutoTokenizer.from_pretrained(baseline_dir / "tokenizer")
    max_length = args.max_length
    validation_frame = frame.loc[frame.Split.eq("validation")].reset_index(drop=True)
    baseline_validation_probs = score_baseline_validation(validation_frame, tokenizer, baseline_dir, max_length)
    baseline_validation = validation_frame[["InChIKey", *ENDPOINTS]].copy()
    for col, endpoint in enumerate(ENDPOINTS):
        baseline_validation[f"{endpoint}_probability"] = baseline_validation_probs[:, col]
    baseline_validation.to_csv(out / "baseline_validation_predictions.csv", index=False)
    dataset = tokenize_with_descriptors(frame, transformed, tokenizer, max_length)
    indices = {s: np.flatnonzero(frame.Split.eq(s).to_numpy()).tolist() for s in SPLITS}
    loaders = {s: DataLoader(Subset(dataset, indices[s]), batch_size=args.batch_size,
                             shuffle=(s == "train"), num_workers=0) for s in SPLITS}
    encoder = AutoModel.from_pretrained(baseline_dir / "encoder")
    model = ChemBERTaDescriptorFusionClassifier(encoder, descriptor_hidden_size=args.descriptor_hidden_size,
                                                 dropout=args.dropout).to(device)
    criterion = MaskedWeightedBCELoss([pos_weights[name] for name in ENDPOINTS]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    environment = env_report(args.seed, device, amp)
    hashes = {}
    for name in ("cleaned_dataset.csv", "split_manifest.csv", "class_distribution_report.csv"):
        hashes[name] = hashlib.sha256((p1_dir / "data" / "processed" / name).read_bytes()).hexdigest()
    config = {"version": 2, "architecture": "ChemBERTa CLS + RDKit descriptor MLP late fusion",
              "baseline_checkpoint": str(baseline_dir), "descriptor_names": list(DESCRIPTOR_NAMES),
              "descriptor_hidden_size": args.descriptor_hidden_size, "dropout": args.dropout,
              "optimizer": "AdamW", "learning_rate": args.learning_rate, "batch_size": args.batch_size,
              "max_epochs": args.epochs, "early_stopping_patience": args.patience,
              "max_length": max_length, "seed": args.seed, "threshold": 0.5,
              "loss": "existing MaskedWeightedBCELoss", "training_only_pos_weights": pos_weights,
              "split_counts": {s: len(indices[s]) for s in SPLITS}, "p1_sha256": hashes,
              "environment": environment}
    _save_json(out / "training_config.json", config)

    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            desc, y = batch["descriptors"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(ids, mask, desc)
                loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, loaders["validation"], criterion, device, amp)
        row = {"epoch": epoch, "train_loss_mean_batch": float(np.mean(train_losses)),
               "validation_loss": validation["loss"], "validation_metrics": validation["endpoints"]}
        history.append(row)
        print(json.dumps({"epoch": epoch, "train_loss": row["train_loss_mean_batch"],
                          "validation_loss": validation["loss"]}), flush=True)
        if np.isfinite(validation["loss"]) and validation["loss"] < best_loss:
            best_loss, best_epoch, wait = validation["loss"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"best_epoch": best_epoch, "best_validation_loss": best_loss,
                        "model_state_dict": best_state}, out / "best_checkpoint.pt")
        else:
            wait += 1
        _save_json(out / "training_history.json", history)
        if wait >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation-loss checkpoint was selected")

    # Only now, after validation-based selection, load the best checkpoint and score test once.
    model.load_state_dict(best_state)
    best_dir = out / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(best_dir / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(best_dir / "tokenizer")
    torch.save(model.state_dict(), best_dir / "model_state_dict.pt")
    model_config = {"architecture": "chemberta_descriptor_late_fusion", "endpoints": list(ENDPOINTS),
                    "descriptor_names": list(DESCRIPTOR_NAMES), "descriptor_hidden_size": args.descriptor_hidden_size,
                    "dropout": args.dropout, "max_length": max_length, "decision_threshold": 0.5,
                    "best_epoch": best_epoch, "best_validation_loss": best_loss}
    _save_json(best_dir / "model_config.json", model_config)
    _save_json(best_dir / "descriptor_preprocessing.json", preprocessor.parameters())
    validation = evaluate(model, loaders["validation"], criterion, device, amp)
    test = evaluate(model, loaders["test"], criterion, device, amp)
    (out / "validation_metrics.json").write_text(json.dumps(
        {"loss": validation["loss"], "endpoints": validation["endpoints"]}, indent=2, allow_nan=False), encoding="utf-8")
    (out / "test_metrics.json").write_text(json.dumps(
        {"loss": test["loss"], "endpoints": test["endpoints"]}, indent=2, allow_nan=False), encoding="utf-8")
    test_rows = frame.loc[frame.Split.eq("test"), ["InChIKey", "Canonical_SMILES", *ENDPOINTS]].reset_index(drop=True)
    test_predictions = test_rows.copy()
    for idx, endpoint in enumerate(ENDPOINTS):
        test_predictions[f"{endpoint}_probability"] = test["probabilities"][:, idx]
        test_predictions[f"{endpoint}_prediction"] = (test["probabilities"][:, idx] >= 0.5).astype(np.int8)
    test_predictions.to_csv(out / "test_predictions.csv", index=False)
    # Compare with the saved V1 validation scoring and previously audited V1 test probabilities.
    comparison = {}
    baseline_evals = {}
    for split, v2_result in (("validation", validation), ("test", test)):
        if split == "validation":
            baseline_frame = baseline_validation
            baseline_probs = baseline_validation_probs
            baseline_labels = validation_frame[list(ENDPOINTS)].to_numpy(dtype=float)
        else:
            v1_path = ROOT / "evaluation" / "chemberta_model_validation" / "test_predictions.csv"
            v1_test = pd.read_csv(v1_path).set_index("InChIKey")
            target_rows = frame.loc[frame.Split.eq("test"), ["InChIKey", *ENDPOINTS]].reset_index(drop=True)
            v1_test = v1_test.loc[target_rows.InChIKey].reset_index()
            if not v1_test.InChIKey.equals(target_rows.InChIKey):
                raise ValueError("Saved V1 test predictions do not align by InChIKey")
            baseline_frame = v1_test
            baseline_probs = np.column_stack([v1_test[f"{ep}_probability"].to_numpy() for ep in ENDPOINTS])
            baseline_labels = target_rows[list(ENDPOINTS)].to_numpy(dtype=float)
        baseline_evals[split] = (baseline_probs.copy(), baseline_labels.copy())
        comparison[split] = {}
        for col, endpoint in enumerate(ENDPOINTS):
            if split == "test":
                v1_label = pd.to_numeric(baseline_frame[f"{endpoint}_label"], errors="coerce").to_numpy(dtype=float)
                if not np.array_equal(np.isnan(v1_label), np.isnan(baseline_labels[:, col])) or not np.array_equal(
                        np.nan_to_num(v1_label, nan=-1), np.nan_to_num(baseline_labels[:, col], nan=-1)):
                    raise ValueError(f"Saved V1 {endpoint} labels differ from frozen P1 labels")
                baseline_labels[:, col] = v1_label
            comparison[split][endpoint] = {
                "v1": _binary_metrics(baseline_labels[:, col], baseline_probs[:, col]),
                "v2": _binary_metrics(v2_result["labels"][:, col], v2_result["probabilities"][:, col]),
            }
    _save_json(out / "baseline_vs_v2_comparison.json", {
        "threshold": 0.5, "test_used_for_model_selection": False,
        "baseline_test_source": "frozen audit test_predictions.csv, aligned and checked by InChIKey",
        "metrics": comparison})
    consistency = {}
    for split, result in (("validation", validation), ("test", test)):
        p = result["probabilities"]
        # baseline_probs and baseline_labels are aligned to the same frozen rows above.
        baseline_p, labels_for_split = baseline_evals[split]
        both = np.isfinite(labels_for_split[:, 1]) & np.isfinite(labels_for_split[:, 2])
        consistency[split] = {
            "rows": int(len(p)),
            "observed_label_hierarchy_violations": int(((labels_for_split[both, 1] == 0) &
                                                         (labels_for_split[both, 2] == 1)).sum()),
            "v1": {"probability_HOB50_gt_HOB20": int((baseline_p[:, 2] > baseline_p[:, 1]).sum()),
                   "binary_HOB50_positive_HOB20_negative_at_0_5": int(((baseline_p[:, 2] >= 0.5) &
                                                                       (baseline_p[:, 1] < 0.5)).sum())},
            "v2": {"probability_HOB50_gt_HOB20": int((p[:, 2] > p[:, 1]).sum()),
                   "binary_HOB50_positive_HOB20_negative_at_0_5": int(((p[:, 2] >= 0.5) &
                                                                       (p[:, 1] < 0.5)).sum())},
        }
    _save_json(out / "hob_probability_consistency.json", consistency)
    report = ["# V2 Descriptor-Fusion Training Comparison", "",
              f"Best checkpoint selected at epoch {best_epoch} by validation masked loss ({best_loss:.6f}).",
              "Test data was evaluated only after this selection. Threshold: 0.5 for both model versions.",
              "Metrics use observed endpoint labels only; PR-AUC is average precision.", "",
              "| Split | Endpoint | Version | N | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | ROC-AUC | PR-AUC |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        for endpoint in ENDPOINTS:
            for version in ("v1", "v2"):
                metric = comparison[split][endpoint][version]
                fmt = lambda key: "n/a" if metric.get(key) is None else f"{metric[key]:.3f}"
                report.append(f"| {split} | {endpoint} | {version.upper()} | {metric.get('n_observed', 0)} | "
                              f"{fmt('accuracy')} | {fmt('precision')} | {fmt('recall_sensitivity')} | "
                              f"{fmt('specificity')} | {fmt('f1')} | {fmt('balanced_accuracy')} | "
                              f"{fmt('roc_auc')} | {fmt('pr_auc_average_precision')} |")
    report.extend(["", "## HOB20/HOB50 consistency", "",
                   "The JSON artifact records V1 and V2 probability-order and binary-threshold violations on the same rows. No hierarchy constraint or post-processing is applied.",
                   "", "## Run configuration", "", f"- Seed: {args.seed}",
                   f"- Optimizer: AdamW; learning rate {args.learning_rate}; batch size {args.batch_size}",
                   f"- Epoch cap / patience: {args.epochs} / {args.patience}; max length {args.max_length}",
                   f"- Environment: {environment}", ""])
    (out / "training_comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps({"training_completed": True,
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "environment": environment}, indent=2), encoding="utf-8")
    return {"best_epoch": best_epoch, "best_validation_loss": best_loss,
            "validation": validation["endpoints"], "test": test["endpoints"]}


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
