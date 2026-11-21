# ChemBERTa Model Validation Report

## Scope and protocol

- Saved evaluation file: outputs/AI_Based_Oral_Absorption_Prediction_System/evaluation.json
- Checkpoint: best_model; unchanged during this audit.
- Classification threshold: 0.5, read from the saved model configuration.
- Saved best epoch: 10; saved best validation loss: 0.35115841031074524.
- Validation and test rows were selected only by the existing split manifest. No new split was created.
- Weights were not selected or adjusted using test results. All test metrics below are post-selection evaluation.
- Missing labels remain NaN and are excluded from only their endpoint's metric calculation.

## Split integrity

- 2289 molecules; 2289 unique InChIKeys; duplicate InChIKey rows: 0.
- One-to-one key set match: True; canonical SMILES match between dataset and manifest: True.
- Molecule keys crossing splits: 0; scaffold groups crossing splits: 0 of 1215.
- Scaffold types: {"Acyclic molecule-specific fallback": 65, "Bemis-Murcko": 2224}.

| Split | Molecules | Share | Unique scaffold groups |
|---|---:|---:|---:|
| train | 1601 | 69.94% | 849 |
| validation | 344 | 15.03% | 183 |
| test | 344 | 15.03% | 183 |

## Endpoint class distributions

Class percentages below use observed labels only. Missing labels are not treated as negative.

| Split | Endpoint | Positive | Negative | Missing | Positive % among observed |
|---|---|---:|---:|---:|---:|
| train | HIA | 693 | 114 | 794 | 85.9% |
| train | HOB20 | 823 | 244 | 534 | 77.1% |
| train | HOB50 | 602 | 487 | 512 | 55.3% |
| validation | HIA | 160 | 14 | 170 | 92.0% |
| validation | HOB20 | 174 | 58 | 112 | 75.0% |
| validation | HOB50 | 126 | 110 | 108 | 53.4% |
| test | HIA | 138 | 40 | 166 | 77.5% |
| test | HOB20 | 154 | 56 | 134 | 73.3% |
| test | HOB50 | 113 | 100 | 131 | 53.1% |

## Validation and test metrics

Confusion matrix convention: [[TN, FP], [FN, TP]]. Sensitivity equals recall. Specificity = TN/(TN+FP). PR-AUC is reported as average precision (AP).

| Split | Endpoint | n observed | Pos / Neg | TN | FP | FN | TP | Sensitivity | Specificity | Precision | F1 | Accuracy | ROC-AUC | PR-AUC (AP) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | HIA | 174 | 160 / 14 | 3 | 11 | 16 | 144 | 0.900 | 0.214 | 0.929 | 0.914 | 0.845 | 0.628 | 0.949 |
| validation | HOB20 | 232 | 174 / 58 | 20 | 38 | 31 | 143 | 0.822 | 0.345 | 0.790 | 0.806 | 0.703 | 0.604 | 0.794 |
| validation | HOB50 | 236 | 126 / 110 | 50 | 60 | 27 | 99 | 0.786 | 0.455 | 0.623 | 0.695 | 0.631 | 0.687 | 0.700 |
| test | HIA | 178 | 138 / 40 | 36 | 4 | 24 | 114 | 0.826 | 0.900 | 0.966 | 0.891 | 0.843 | 0.938 | 0.977 |
| test | HOB20 | 210 | 154 / 56 | 31 | 25 | 50 | 104 | 0.675 | 0.554 | 0.806 | 0.735 | 0.643 | 0.685 | 0.834 |
| test | HOB50 | 213 | 113 / 100 | 66 | 34 | 38 | 75 | 0.664 | 0.660 | 0.688 | 0.676 | 0.662 | 0.687 | 0.703 |

## Test false positives and false negatives

| Endpoint | False positives | False negatives |
|---|---:|---:|
| HIA | 4 | 24 |
| HOB20 | 25 | 50 |
| HOB50 | 34 | 38 |

Per-compound InChIKeys, canonical SMILES, labels, probabilities, and predictions are in test_false_positives_and_negatives.csv. Only observed labels appear in this error file.

## HOB20/HOB50 nested-threshold consistency

- {"interpretation": "Conditional nesting check: if both labels mark the same oral-bioavailability quantity at 20% and 50% cutoffs, HOB50 positive with HOB20 negative is inconsistent.", "test_rows": 344, "test_binary_prediction_violations": 3, "test_probability_order_violations_HOB50_gt_HOB20": 79, "test_rows_with_both_observed_labels": 210, "test_observed_label_violations": 0, "full_data_rows_with_both_observed_labels": 1509, "full_data_observed_label_violations": 0, "test_prediction_violation_rows": [{"InChIKey": "AVKUERGKIZMTKX-NJBDSQKTSA-N", "Canonical_SMILES": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O", "HOB20_probability": 0.4731251895427704, "HOB20_prediction": 0, "HOB50_probability": 0.507572591304779, "HOB50_prediction": 1}, {"InChIKey": "AVKUERGKIZMTKX-OXIWPEFWSA-N", "Canonical_SMILES": "CC1(C)S[C@H]2[C@H](NC(=O)[C@@H](N)c3ccccc3)C(=O)N2[C@@H]1C(=O)O", "HOB20_probability": 0.4731251895427704, "HOB20_prediction": 0, "HOB50_probability": 0.507572591304779, "HOB50_prediction": 1}, {"InChIKey": "YLJREFDVOIBQDA-UHFFFAOYSA-N", "Canonical_SMILES": "N=c1c2c([nH]c3ccccc13)CCCC2", "HOB20_probability": 0.4994474947452545, "HOB20_prediction": 0, "HOB50_probability": 0.5015698075294495, "HOB50_prediction": 1}]}
- This is a consistency diagnostic only; no predictions were changed or post-processed.
- Interpretation assumes the two endpoints measure the same underlying bioavailability quantity at nested cutoffs. That source definition has not been independently confirmed.

## Reproduction against saved evaluation

The same root best checkpoint was compared with the saved validation/test metrics. Differences are:

| Split / endpoint | Max absolute metric difference | Confusion matrix matched |
|---|---:|---|
| validation / HIA | 0.000446429 | True |
| validation / HOB20 | 1.11022e-16 | True |
| validation / HOB50 | 0.0001443 | True |
| test / HIA | 1.11022e-16 | True |
| test / HOB20 | 0.000115955 | True |
| test / HOB50 | 1.11022e-16 | True |

## Leakage, bias, and generalization limitations

- The split is InChIKey-unique and scaffold-disjoint across train, validation, and test. This supports an internal scaffold-split assessment, not an independent external benchmark.
- The P1 quality report says original benchmark/test-sheet compound identities were unavailable. Overlap with those original benchmark compounds cannot be checked from current files.
- Endpoint metrics use different observed subsets because labels are missing at different rates. Results across endpoints are not paired on a shared cohort.
- Test supports are endpoint-specific and limited; class-conditional estimates can have substantial sampling uncertainty. No confidence intervals are included.
- HIA and HOB20 are class-imbalanced in training and test. Training class weights do not correct source, assay, or chemical-domain bias.
- The P1 report records HIA positive as >30%, but the authoritative source definition is not present here; >30% versus >=30% remains unresolved. Labels were not changed.
- No nutraceutical-specific identity, cohort, or external test set is present. Internal scaffold-split results do not support generalization claims specifically for nutraceutical compounds.
- Saved evaluation JSON has no per-epoch history or probability-level rows. This audit recomputed validation/test predictions with the identical saved weights.

## Generated artifacts

- metrics_by_endpoint.csv: validation and test metrics and endpoint support.
- split_integrity.json: frozen split checks and distribution counts.
- test_predictions.csv: test labels (NaNs retained), probabilities, and predictions.
- test_false_positives_and_negatives.csv: observed-label mistakes by endpoint.
- test_confusion_matrices.svg and test_precision_recall_curves.svg: plots.

No model, dataset, split, training, or inference files were modified.
