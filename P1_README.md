# AI-Based Oral Absorption Prediction System

P1 handoff

P1 creates the clean structural table, frozen Bemis–Murcko scaffold split,
split-wise class distribution, training-only positive weights, reusable masked
loss, quality report, and audit files used by P2/P3/P4. It does not train a
prediction model.

## Run the pipeline

From the project root:

```powershell
python run_p1_pipeline.py
```

The source defaults to `data/processed/oralabs_model_training.csv`. To use a
different compatible source or seed:

```powershell
python run_p1_pipeline.py --input data/processed/oralabs_model_training.csv --seed 42
```

The pipeline never overwrites the input CSV. It writes the following under
`data/processed/`:

- `cleaned_dataset.csv`: original and canonical SMILES, computed InChIKey,
  input key audit fields, labels, and source row tracking.
- `invalid_records.csv`: rows excluded because the structure was missing or
  RDKit could not parse/sanitize it.
- `duplicate_audit.csv`: duplicate structures and their resolution.
- `label_conflict_audit.csv`: per endpoint conflicts and the resolution.
- `split_manifest.csv`: frozen molecule/scaffold assignment.
- `class_distribution_report.csv`: endpoint counts for each split; weights are
  populated only for the training rows.
- `data_quality_report.md`: computed statistics, limitations, and integration
  example.

The three independent stages can also be run in order:

```powershell
python data_preprocessing.py
python scaffold_split.py --seed 42
python class_weights.py
```

## Shared model interface

Join `cleaned_dataset.csv` and `split_manifest.csv` one-to-one on `InChIKey`.
Use the manifest `Split` field; do not create a separate split in another model
branch. Use endpoint order `HIA`, `HOB20`, `HOB50`. Keep missing values as NaN
and build a same-shaped mask. `InChIKey` identifies rows; it is not a model
feature.

The `Train_Pos_Weight` column stores `N_negative / N_positive`, calculated only
from observed labels in the training split. If either training class is absent,
the weight is empty and the report states why. Do not replace it silently.

Example:

```python
import pandas as pd
import torch
from masked_weighted_loss import MaskedWeightedBCELoss

data = pd.read_csv("data/processed/cleaned_dataset.csv")
manifest = pd.read_csv("data/processed/split_manifest.csv")
train_ids = set(manifest.loc[manifest.Split == "train", "InChIKey"])
train = data[data.InChIKey.isin(train_ids)].reset_index(drop=True)

weight_table = pd.read_csv("data/processed/class_distribution_report.csv")
train_weights = weight_table.query("Split == 'train'").set_index("Endpoint")
weights = [float(train_weights.loc[name, "Train_Pos_Weight"])
           for name in ("HIA", "HOB20", "HOB50")]
criterion = MaskedWeightedBCELoss(weights)

targets = torch.as_tensor(
    train[["HIA", "HOB20", "HOB50"]].to_numpy(dtype="float32")
)  # NaN remains missing
logits = model_output  # raw [batch, 3] logits; no sigmoid before the loss
loss, per_endpoint_losses = criterion(logits, targets, return_details=True)
```

For RDKit descriptors, fit imputation and scaling using only training rows, then
apply the fitted transform to validation and test. Fingerprint bits stay binary.
ChemBERTa and RDKit models must use the same row order, split manifest, target
order, and missing-label mask.

## Validate the shared components

```powershell
python -m unittest discover -s tests -v
```

The checks cover invalid structures, canonicalization, duplicate/conflict
auditing, deterministic scaffold grouping, training-only weights, masked loss,
finite loss, and gradient propagation.
