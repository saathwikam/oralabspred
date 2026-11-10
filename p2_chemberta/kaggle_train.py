from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
PROJECT_NAME = "AI-Based Oral Absorption Prediction System"
PROJECT_FOLDER = "AI_Based_Oral_Absorption_Prediction_System"
OUTPUT_DIR = Path("/kaggle/working") / PROJECT_FOLDER


def locate_p1_root() -> Path:
    # Kaggle flattens a dataset's single top-level directory during upload, but
    # local bundles or manually attached copies may retain that project folder.
    candidates = sorted(
        {
            path.parents[2]
            for dataset_dir in INPUT_ROOT.iterdir()
            if dataset_dir.is_dir()
            for path in dataset_dir.rglob("cleaned_dataset.csv")
            if path.parent.name == "processed" and path.parent.parent.name == "data"
        }
    ) if INPUT_ROOT.is_dir() else []
    if len(candidates) != 1:
        mounted_datasets = sorted(path.name for path in INPUT_ROOT.iterdir()) if INPUT_ROOT.is_dir() else []
        raise RuntimeError(
            "Expected exactly one attached Kaggle dataset containing "
            "data/processed/cleaned_dataset.csv; found "
            f"{len(candidates)}. Mounted /kaggle/input entries: {mounted_datasets}. "
            "Attach the private AI-Based Oral Absorption Prediction System bundle."
        )
    return candidates[0]


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if importlib.util.find_spec("transformers") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "transformers>=4.44,<6"],
            check=True,
        )

    import torch

    p1_dir = locate_p1_root()
    train_script = p1_dir / "p2_chemberta" / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"P2 trainer not found in attached bundle: {train_script}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"P1 bundle: {p1_dir}")
    print(f"Device detected: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Outputs: {OUTPUT_DIR}")

    command = [
        sys.executable,
        str(train_script),
        "--p1-dir", str(p1_dir),
        "--output-dir", str(OUTPUT_DIR),
        "--checkpoint", "DeepChem/ChemBERTa-77M-MLM",
        "--learning-rate", "2e-5",
        "--batch-size", "16",
        "--epochs", "10",
        "--patience", "3",
        "--max-length", "256",
        "--mixed-precision",
    ]
    local_checkpoint = OUTPUT_DIR / "last_checkpoint.pt"
    if local_checkpoint.is_file():
        command.extend(["--resume-from", str(local_checkpoint)])
    else:
        input_checkpoints = sorted(INPUT_ROOT.rglob("last_checkpoint.pt"))
        if len(input_checkpoints) == 1:
            command.extend(["--resume-from", str(input_checkpoints[0])])
        elif len(input_checkpoints) > 1:
            raise RuntimeError("More than one resume checkpoint is attached; keep only the intended checkpoint.")

    log_path = OUTPUT_DIR / "training.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n=== AI-Based Oral Absorption Prediction System Kaggle run ===\n")
        log.write(f"Command: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=str(train_script.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        result = process.wait()
        if result != 0:
            raise subprocess.CalledProcessError(result, command)
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
