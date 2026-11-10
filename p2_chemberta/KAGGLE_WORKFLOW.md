# AI-Based Oral Absorption Prediction System - Kaggle training workflow

This workflow stages copies of the required P1 files and reuses `data.py`, `model.py`, and `train.py`. It keeps the dataset and kernel private, enables GPU/internet in the kernel metadata, and writes outputs to `/kaggle/working/AI_Based_Oral_Absorption_Prediction_System`.

## One-time Kaggle setup

1. Create/sign in to Kaggle and complete phone verification. Kaggle requires phone verification for GPU access; free GPU availability and weekly quota can vary.
2. Install the official Kaggle CLI with Python 3.11 or later, then run `kaggle auth login`. Complete the browser authorization locally. Do not paste credentials into chat or save them in project files. The legacy `kaggle.json` method is also supported, but keep that file outside the repository.
3. Confirm Kaggle Notebook GPU and Internet access are enabled for the account. The model checkpoint is downloaded from Hugging Face at runtime.

## Prepare and upload, from local PowerShell

```powershell
py -3.13 -m pip install kaggle
kaggle auth login

$P1 = 'C:\Users\macha\OneDrive\Desktop\PS\AI_Based_Oral_Absorption_Prediction_System'
$Bundle = 'C:\Users\macha\OneDrive\Desktop\PS\AI_Based_Oral_Absorption_Prediction_System_Kaggle_Bundle'
$KaggleUser = 'YOUR_KAGGLE_USERNAME'

py -3.13 "$P1\p2_chemberta\prepare_kaggle.py" `
  --p1-dir $P1 `
  --bundle-dir $Bundle `
  --username $KaggleUser

kaggle datasets create -p "$Bundle\dataset" -r zip -t
kaggle kernels push -p "$Bundle\kernel" --no-run
```

The bundle contains copies of only `cleaned_dataset.csv`, `split_manifest.csv`, `class_distribution_report.csv`, `masked_weighted_loss.py`, and the three P2 implementation files. The staging script refuses to write into the P1 directory. The Kaggle dataset is private by default; metadata uses Kaggle's `unknown` license value because the source license has not been established.

## Start training from local

After confirming the private dataset was created and the no-run kernel version exists, start training with:

```powershell
kaggle kernels push -p "$Bundle\kernel"
```

Kaggle's `kernels push` runs the kernel unless `--no-run` is supplied. This final command starts the full training run. Training was not started during preparation.

## Outputs and resume

The launcher writes `best_model/` (encoder, tokenizer, task-head state, model config, and training config), `training_history.json`, `evaluation.json`, `training.log`, and `last_checkpoint.pt` under `/kaggle/working/AI_Based_Oral_Absorption_Prediction_System`. The checkpoint is replaced atomically at each completed epoch and includes optimizer/scaler state, early-stopping state, history, and RNG state. Re-running the same Kaggle session auto-resumes it; a checkpoint attached under `/kaggle/input/*/last_checkpoint.pt` is also auto-detected. P1 file hashes are checked before resuming so a checkpoint cannot silently continue against different data or splits.

To download completed kernel outputs locally:

```powershell
kaggle kernels output $KaggleUser/ai-based-oral-absorption-prediction-system -p "$Bundle\outputs"
```

If Kaggle discards the working directory when a session is terminated, attach the recovered `last_checkpoint.pt` as a private Kaggle Dataset to the kernel before rerunning. Kaggle preserves committed kernel outputs, while `/kaggle/working` is the writable output area; an interrupted, uncommitted session may need its latest checkpoint restored explicitly.
