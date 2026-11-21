# Version 2 Descriptor-Fusion Experiment Report

## Status

The separate V2 implementation, descriptor audit, and smoke/unit tests are complete. **No V2 model training has been run**, so there are no V2 performance results and no claim of improvement. The environment has PyTorch `2.7.0+cpu`; `torch.cuda.is_available()` is false. Full ChemBERTa fine-tuning should therefore be run on Kaggle GPU using the instructions in `README.md`. The existing V1 checkpoint and all P1 inputs were preserved.

## Existing baseline and study design

The frozen split contains 1,601 training, 344 validation, and 344 test compounds. V2 preserves these assignments and the training-only positive weights. Training is configured for validation masked-loss checkpoint selection; the test split is evaluated only after the best validation checkpoint has been selected. The decision threshold remains 0.5 for this experiment.

Validation-only V1 reference values from the saved validation probabilities:

| Endpoint | Observed n (positive / negative) | F1 at 0.5 | Precision | Recall | Specificity | Balanced accuracy | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HIA | 174 (160 / 14) | 0.914 | 0.929 | 0.900 | 0.214 | 0.557 | 0.628 | 0.949 |
| HOB20 | 232 (174 / 58) | 0.806 | 0.790 | 0.822 | 0.345 | 0.583 | 0.604 | 0.794 |
| HOB50 | 236 (126 / 110) | 0.695 | 0.623 | 0.786 | 0.455 | 0.620 | 0.687 | 0.700 |

HOB20 and HOB50 discrimination is the primary motivation. Their validation ROC-AUC/AP should be compared against these values after V2 training; any threshold-only effects are excluded because V2 evaluation holds 0.5 fixed.

## Descriptor audit

RDKit generated all six requested descriptors for all 2,289 compounds: **2,289 rows retained; 0 descriptor calculation failures**. Descriptor transformations were fit on the 1,601 training rows only. Validation/test rows were transformed using those fitted training statistics. The preprocessing JSON records the medians, scaler means/scales, fit split, and P1 data/manifest SHA-256 hashes.

Raw/scaled descriptor values and the failure record are stored in this folder's `run_artifacts/`. Missing descriptor values, if any occur in a future input, are explicitly audited and median-imputed using training-only medians; no input compound is dropped.

## Implementation and verification

- New architecture: V1 fine-tuned ChemBERTa encoder + six-value RDKit MLP (32 hidden units, ReLU/dropout) + concatenated representation + three independent endpoint heads.
- Existing masked weighted BCE, training-only class weights, endpoint masks, and frozen split are retained.
- No hierarchy loss or prediction post-processing is included. Training outputs will compare HOB probability ordering and binary nesting against V1 on matched validation/test rows.
- Five focused unit tests pass for descriptor values/failure audit, training-only preprocessing, fused forward output, masked loss, and input alignment.
- Training CLI help/import check succeeds. No full epochs were run because the available environment is CPU-only; no real V2 comparison can be reported yet.

## Artifacts

- `model.py`, `data.py`, `descriptors.py`: isolated V2 implementation.
- `train.py`: validation-selected training and post-selection V1 comparison workflow.
- `prepare_descriptors.py`: full dataset descriptor audit and training-only preprocessing export.
- `README.md`: local and Kaggle GPU instructions.
- `../../../../tests/test_v2_descriptor_fusion.py`: focused unit/smoke tests.
- `run_artifacts/descriptor_audit.json`, `descriptor_values.csv`, `descriptor_values_scaled.csv`, `descriptor_failures.csv`, and `descriptor_preprocessing.json`: actual preprocessing audit outputs.

After a Kaggle run, the training script writes `training_comparison_report.md`, per-endpoint validation/test metrics, training history/configuration/environment, and a separate `best_model/` into the specified V2 output directory.
