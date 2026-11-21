from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data import ENDPOINTS, SPLITS, load_p1_data, tokenize_frame
from model import ChemBERTaMultiTaskClassifier, DEFAULT_CHECKPOINT, load_saved_model


P1_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AI-Based Oral Absorption Prediction System ChemBERTa multi-task classifier.")
    parser.add_argument("--p1-dir", type=Path, default=P1_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-auto-resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_environment(seed: int, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    try:
        import rdkit

        rdkit_version: str | None = rdkit.__version__
    except ImportError:
        rdkit_version = None
    gpu_name: str | None = None
    gpu_memory: dict[str, float] | None = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gpu_memory = {
            "available_gib_at_start": free_bytes / (1024**3),
            "total_gib": total_bytes / (1024**3),
        }
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "rdkit_version": rdkit_version,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
        "random_seed": int(seed),
        "mixed_precision_enabled": bool(amp_enabled),
    }


def cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Snapshot model weights on CPU so subsequent optimizer steps cannot mutate them."""
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def evaluate(
    model: ChemBERTaMultiTaskClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.float().cpu())
            all_targets.append(batch["labels"].cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    loss = float(criterion(logits, targets).item())
    probabilities = torch.sigmoid(logits).numpy()
    target_array = targets.numpy()
    reports: dict[str, Any] = {}
    for column, endpoint in enumerate(ENDPOINTS):
        mask = np.isfinite(target_array[:, column])
        y_true = target_array[mask, column].astype(np.int32)
        y_prob = probabilities[mask, column]
        if y_true.size == 0:
            reports[endpoint] = {
                "n_observed": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "confusion_matrix_labels_0_1": [[0, 0], [0, 0]],
            }
            continue
        y_pred = (y_prob >= 0.5).astype(np.int32)
        try:
            auc: float | None = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = None
        reports[endpoint] = {
            "n_observed": int(y_true.size),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix_labels_0_1": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }
    return loss, reports


def save_best(
    model: ChemBERTaMultiTaskClassifier,
    tokenizer: Any,
    output_dir: Path,
    checkpoint: str,
    max_length: int,
    training_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
    model_config = {
        "project_name": "AI-Based Oral Absorption Prediction System",
        "architecture": "shared_chemberta_encoder_three_binary_heads",
        "checkpoint": checkpoint,
        "endpoints": list(ENDPOINTS),
        "hidden_size": int(model.encoder.config.hidden_size),
        "dropout": float(model.dropout.p),
        "max_length": int(max_length),
        "decision_threshold": 0.5,
    }
    (output_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, max-length, and patience must be positive.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    output_dir = args.output_dir.resolve()
    p1_dir = args.p1_dir.resolve()
    try:
        output_dir.relative_to(p1_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Model output directory must be outside the P1 directory to preserve P1 files.")
    frame, pos_weights = load_p1_data(p1_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    datasets = {
        split: tokenize_frame(frame.loc[frame["Split"].eq(split)].reset_index(drop=True), tokenizer, args.max_length)
        for split in SPLITS
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    encoder = AutoModel.from_pretrained(args.checkpoint)
    model = ChemBERTaMultiTaskClassifier(encoder).to(device)
    p1_dir_text = str(p1_dir)
    sys_path = str(p1_dir)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from masked_weighted_loss import MaskedWeightedBCELoss

    criterion = MaskedWeightedBCELoss([pos_weights[name] for name in ENDPOINTS]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    current_environment = training_environment(args.seed, device, amp_enabled)
    training_config: dict[str, Any] = {
        "project_name": "AI-Based Oral Absorption Prediction System",
        "checkpoint": args.checkpoint,
        "p1_dir": p1_dir_text,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "max_length": args.max_length,
        "seed": args.seed,
        "mixed_precision_requested": bool(args.mixed_precision),
        "mixed_precision_enabled": amp_enabled,
        "device": str(device),
        "training_environment": current_environment,
        "training_sessions": [current_environment],
        "endpoints": list(ENDPOINTS),
        "training_only_pos_weights": pos_weights,
        "split_manifest": str(p1_dir / "data" / "processed" / "split_manifest.csv"),
    }
    processed_dir = p1_dir / "data" / "processed"
    file_hashes = {}
    for filename in ("cleaned_dataset.csv", "split_manifest.csv", "class_distribution_report.csv"):
        digest = hashlib.sha256((processed_dir / filename).read_bytes()).hexdigest()
        file_hashes[filename] = digest
    resume_signature = {
        "checkpoint": args.checkpoint,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "max_length": args.max_length,
        "seed": args.seed,
        "mixed_precision_requested": bool(args.mixed_precision),
        "file_hashes": file_hashes,
    }

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_model_state: dict[str, torch.Tensor] | None = None
    checkpoint_path = output_dir / "last_checkpoint.pt"
    resume_path = args.resume_from
    if resume_path is None and not args.no_auto_resume and checkpoint_path.is_file():
        resume_path = checkpoint_path
    if resume_path is not None:
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume_state.get("resume_signature") != resume_signature:
            raise ValueError("Resume checkpoint configuration or P1 data/split hashes do not match this run.")
        previous_config = resume_state.get("training_config")
        if previous_config is not None:
            training_config["training_environment"] = previous_config.get(
                "training_environment", current_environment
            )
            sessions = previous_config.get("training_sessions") or [
                training_config["training_environment"]
            ]
            training_config["training_sessions"] = [*sessions, current_environment]
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_epoch = int(resume_state["completed_epoch"]) + 1
        best_validation_loss = float(resume_state["best_validation_loss"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        history = list(resume_state["history"])
        saved_best_state = resume_state.get("best_model_state_dict")
        if saved_best_state is not None:
            best_model_state = cpu_state_dict(saved_best_state)
        else:
            # Backward compatibility for older checkpoints whose best model is
            # still present in the same output directory.
            legacy_best_path = output_dir / "best_model" / "model_state_dict.pt"
            if legacy_best_path.is_file():
                best_model_state = cpu_state_dict(
                    torch.load(legacy_best_path, map_location="cpu", weights_only=True)
                )
            elif best_epoch == int(resume_state["completed_epoch"]):
                best_model_state = cpu_state_dict(resume_state["model_state_dict"])
            else:
                raise ValueError(
                    "This legacy resume checkpoint does not contain best validation weights, "
                    "and no saved best_model is available. Resume from the original output "
                    "directory or use a checkpoint created by this version."
                )
        torch.set_rng_state(resume_state["torch_rng_state"])
        if device.type == "cuda" and resume_state.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])
        print(f"Resuming after epoch {start_epoch - 1} from {resume_path}")
    else:
        print("Starting a new training run.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().item()))

        validation_loss, _ = evaluate(model, loaders["validation"], criterion, device, amp_enabled)
        if not np.isfinite(validation_loss):
            raise RuntimeError("Validation loss is not finite; refusing to save a best model.")
        record = {
            "epoch": epoch,
            "train_loss_mean_batch": float(np.mean(train_losses)) if train_losses else None,
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(json.dumps(record))
        stop_early = False
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_model_state = cpu_state_dict(model.state_dict())
            save_best(model, tokenizer, output_dir / "best_model", args.checkpoint, args.max_length, training_config)
        else:
            epochs_without_improvement += 1
            stop_early = epochs_without_improvement >= args.patience

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        resume_state = {
            "resume_signature": resume_signature,
            "completed_epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "best_model_state_dict": best_model_state,
            "training_config": training_config,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(resume_state, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
        if stop_early:
            break

    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if best_model_state is None:
        raise RuntimeError("No best validation model weights were captured; refusing test evaluation.")
    model.load_state_dict(best_model_state)
    save_best(model, tokenizer, output_dir / "best_model", args.checkpoint, args.max_length, training_config)
    best_model, _, _ = load_saved_model(output_dir / "best_model", device=device)
    evaluation: dict[str, Any] = {"project_name": "AI-Based Oral Absorption Prediction System", "best_epoch": best_epoch, "best_validation_loss": best_validation_loss}
    for split in ("validation", "test"):
        split_loss, split_metrics = evaluate(best_model, loaders[split], criterion, device, amp_enabled)
        evaluation[split] = {"loss": split_loss, "endpoints": split_metrics}
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "evaluation": evaluation}, indent=2))
    return evaluation


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
