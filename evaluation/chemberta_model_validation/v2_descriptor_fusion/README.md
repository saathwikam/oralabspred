# Version 2: ChemBERTa + RDKit descriptor late fusion

This experiment is isolated from Version 1. It reads P1 data, the frozen manifest and training-only class weights but never writes to them. V1 model/inference files and production thresholds are unchanged.

## Architecture and data

The V1 best encoder initializes a shared ChemBERTa CLS embedding. Six RDKit descriptors (`MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds`) pass through a 32-unit ReLU/dropout branch. The representations are concatenated and feed three independent binary heads. No HOB hierarchy constraint or probability post-processing is used. The existing `MaskedWeightedBCELoss` and frozen P1 `Train_Pos_Weight` values are retained; missing targets stay NaN and remain masked.

Descriptor medians and scaling statistics are fit only on `Split=train`, then applied to all partitions. Failed calculations are retained as missing descriptor vectors and written to an explicit failure audit. The actual dataset audit found zero descriptor failures.

## Local prepare and train commands

From the project root:

```powershell
python evaluation/chemberta_model_validation/v2_descriptor_fusion/prepare_descriptors.py
python evaluation/chemberta_model_validation/v2_descriptor_fusion/train.py --p1-dir . --baseline-dir best_model --output-dir evaluation/chemberta_model_validation/v2_descriptor_fusion/run_artifacts
```

Defaults: AdamW, learning rate 2e-5, batch size 16, maximum 10 epochs, validation-loss early stopping patience 3, maximum token length 256, seed 42, 32 descriptor hidden units, dropout 0.1, and mixed precision only when CUDA is available. The best checkpoint is selected strictly by validation masked loss. Test metrics are computed only after selection and never used for thresholds or model choices.

## Kaggle GPU workflow

1. Create a Kaggle dataset containing the required project files: `data/processed/cleaned_dataset.csv`, `data/processed/split_manifest.csv`, `data/processed/class_distribution_report.csv`, `masked_weighted_loss.py`, `p2_chemberta/model.py`, `evaluation/chemberta_model_validation/test_predictions.csv` (saved V1 test probabilities for matched post-selection comparison), the six `best_model/` artifacts, and this V2 package. Add that dataset to a Kaggle Notebook and enable its GPU accelerator.
2. In a notebook cell, replace `<your-dataset-folder>` with Kaggle's mounted dataset directory:

```python
from pathlib import Path
import sys

source = Path("/kaggle/input/<your-dataset-folder>")
sys.path.insert(0, str(source / "evaluation/chemberta_model_validation/v2_descriptor_fusion"))
from prepare_descriptors import prepare
prepare(p1_dir=source, output_dir=Path("/kaggle/working/v2_descriptor_preparation"))
```

3. Run training in another cell:

```bash
!python /kaggle/input/<your-dataset-folder>/evaluation/chemberta_model_validation/v2_descriptor_fusion/train.py --p1-dir /kaggle/input/<your-dataset-folder> --baseline-dir /kaggle/input/<your-dataset-folder>/best_model --output-dir /kaggle/working/v2_descriptor_fusion
```

The script detects CUDA and enables AMP only on GPU; CPU is supported but full fine-tuning is expected to be substantially slower. Save `/kaggle/working/v2_descriptor_fusion` as a Kaggle notebook output/version artifact after the run. Never point `--output-dir` at the V1 model directory.

## Artifacts

`prepare_descriptors.py` writes raw/scaled descriptor tables, a calculation-failure log, source hashes, and fitted imputation/scaling parameters. Training writes a separate output directory containing best V2 weights, encoder/tokenizer, training configuration/environment, history, validation metrics, final test metrics, V1-versus-V2 comparison, test predictions, and HOB consistency diagnostics.
