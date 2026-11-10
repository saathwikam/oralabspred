# AI-Based Oral Absorption Prediction System - P1 Dataset Quality Report

## Source and reproducibility

- Source CSV: `oralabs_model_training.csv`
- Source CSV SHA-256: `b39a8a67957a53b6be63b0f64b81a579e4f74b30710019dbdb2d02830aa22e02`
- Random seed: `42`
- Requested split fractions: train 70.00%, validation 15.00%, test 15.00%
- Python: `3.13.7`
- RDKit: `2026.09.1`; pandas: `2.3.2`; NumPy: `2.2.5`; PyTorch: `2.10.0+cpu`

## Dataset and structure checks

- Original dimensions: 2289 rows × 5 columns (`InChIKey, SMILES, HIA, HOB20, HOB50`).
- Valid SMILES: 2289; invalid or missing SMILES: 0.
- Unique molecules after computed-InChIKey deduplication: 2289.
- Duplicate structure groups: 0; extra duplicate rows coalesced: 0.
- Conflicting duplicate endpoint labels: 0; those endpoint values were set missing and audited.
- Input/computed InChIKey mismatches: 0.
- Invalid records: `invalid_records.csv`; duplicate audit: `duplicate_audit.csv`; label conflict audit: `label_conflict_audit.csv`.

### Endpoint meanings and cleaned missing counts

- **HIA:** Positive (1) means source HIA percentage > 30%; negative (0) means <= 30%.
- **HOB20:** Positive (1) means measured oral bioavailability value >= 20%; negative (0) is below 20%.
- **HOB50:** Positive (1) means measured oral bioavailability value >= 50%; negative (0) is below 50%.
- Missing `HIA` labels in original source: 1130.
- Missing `HOB20` labels in original source: 780.
- Missing `HOB50` labels in original source: 751.
- Missing `HIA` labels after cleaning: 1130 / 2289.
- Missing `HOB20` labels after cleaning: 780 / 2289.
- Missing `HOB50` labels after cleaning: 751 / 2289.

## Scaffold split

| Split | Molecules | Actual proportion | Unique scaffold groups |
|---|---:|---:|---:|
| train | 1601 | 69.94% | 849 |
| validation | 344 | 15.03% | 183 |
| test | 344 | 15.03% | 183 |

- Unique scaffold groups: 1215; acyclic fallback molecules: 65.
- Scaffold leakage groups across splits: 0.
- Molecule-level and scaffold-level split assignment is frozen in `split_manifest.csv`.

## Endpoint distributions and class weights

`pos_weight = N_negative / N_positive`, calculated from observed training labels only. Validation/test rows have no training weight.

| Endpoint | Split | Positive | Negative | Missing | Positive % | Negative % | Training pos_weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIA | train | 693 | 114 | 794 | 85.87% | 14.13% | 0.164502 |
| HIA | validation | 160 | 14 | 170 | 91.95% | 8.05% | — |
| HIA | test | 138 | 40 | 166 | 77.53% | 22.47% | — |
| HOB20 | train | 823 | 244 | 534 | 77.13% | 22.87% | 0.296476 |
| HOB20 | validation | 174 | 58 | 112 | 75.00% | 25.00% | — |
| HOB20 | test | 154 | 56 | 134 | 73.33% | 26.67% | — |
| HOB50 | train | 602 | 487 | 512 | 55.28% | 44.72% | 0.808970 |
| HOB50 | validation | 126 | 110 | 108 | 53.39% | 46.61% | — |
| HOB50 | test | 113 | 100 | 131 | 53.05% | 46.95% | — |

## Known limitations

- The model table has partially observed endpoints; blank values are unassayed, not negative, and must remain masked during training.
- Class weighting addresses class imbalance only; it does not correct source, assay, or chemical-domain bias.
- Scaffold grouping can make actual split proportions and endpoint class proportions differ from the requested fractions.
- The five-column model CSV has no source-row/sheet membership. This run does not exclude compounds from the source workbook's original HOB test sheets from the new scaffold split; reconcile source identities before claiming an independent external benchmark evaluation.
- The 65 acyclic molecules use molecule-specific fallback groups because their Bemis-Murcko scaffold is empty; identical structures remain grouped by computed InChIKey.

## Integration interface for P2/P3/P4

Use `cleaned_dataset.csv` and `split_manifest.csv`; never make a new row-wise split in a model branch. Join one-to-one on `InChIKey`. Read target columns in `HIA`, `HOB20`, `HOB50` order and preserve NaN values. Select train/validation/test using the manifest's `Split` column. Load the three training `Train_Pos_Weight` values from `class_distribution_report.csv` in endpoint order. If a training weight is undefined because a class has no observed examples, do not silently substitute a value; report that task as not trainable with class balancing.

```python
import pandas as pd
import torch
from masked_weighted_loss import MaskedWeightedBCELoss

data = pd.read_csv('data/processed/cleaned_dataset.csv')
manifest = pd.read_csv('data/processed/split_manifest.csv')
train_ids = set(manifest.loc[manifest.Split == 'train', 'InChIKey'])
train = data[data.InChIKey.isin(train_ids)].reset_index(drop=True)
weight_table = pd.read_csv('data/processed/class_distribution_report.csv')
weight_rows = weight_table.query("Split == 'train'").set_index('Endpoint')
pos_weights = [float(weight_rows.loc[name, 'Train_Pos_Weight']) for name in ('HIA', 'HOB20', 'HOB50')]
criterion = MaskedWeightedBCELoss(pos_weights)
targets = torch.as_tensor(train[['HIA', 'HOB20', 'HOB50']].to_numpy(dtype='float32'))  # NaN stays missing
logits = model_output  # raw [batch, 3] logits; do not apply sigmoid before this loss
loss, per_task = criterion(logits, targets, return_details=True)
```

For RDKit descriptors, fit imputation/scaling on training rows only and reuse the fitted transform for validation and test. ChemBERTa and RDKit branches must consume the same molecule order and frozen split.
