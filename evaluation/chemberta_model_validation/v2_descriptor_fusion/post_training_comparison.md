# Version 1 vs Version 2 Post-Training Comparison

## Executive summary

The recovered Version 2 artifacts are internally consistent and align with the frozen P1 split. V2's best checkpoint is epoch 4, selected at minimum validation masked loss 0.3517275751; its state tensors exactly match the saved `best_model/model_state_dict.pt`. The V1 and V2 test prediction files contain the same 344 unique InChIKeys, exactly the frozen test assignments, and endpoint labels/missingness identical to P1.

Descriptor fusion does **not** show a consistent improvement for HOB20 or HOB50. Test ROC-AUC increases slightly for both, but paired 95% bootstrap intervals include zero; F1 and accuracy are lower for both. Validation results are mixed and do not consistently support the test-set ranking changes. HOB binary contradictions decrease, but probability inversions increase. Keep V1 as the production baseline; V2 remains an exploratory candidate, not a demonstrated replacement.

## Artifact inspection and provenance

The recovered folder contains JSON, CSV, Markdown and PyTorch checkpoint files, plus the V2 encoder/tokenizer and model configuration. The prediction artifacts are CSVs with row-level identifiers, observed labels and endpoint probabilities. The metrics and histories are JSON. `best_checkpoint.pt` loads as a PyTorch weights-only dictionary containing `best_epoch`, `best_validation_loss`, and `model_state_dict`.

Checkpoint checks:

- `best_checkpoint.pt`: epoch 4; best validation loss 0.35172757506370544.
- `best_model/model_config.json`: epoch 4; the same best validation loss; decision threshold 0.5.
- `run_summary.json`: epoch 4; the same best validation loss; training completed on a Tesla T4 with CUDA/mixed precision.
- The checkpoint and saved best-model state dicts each have 63 matching keys; every tensor value is identical.
- Training config records seed 42, AdamW, learning rate 2e-5, batch 16, maximum 10 epochs, patience 3, max length 256, and the existing training-only weights. Its SHA-256 values for the cleaned data, frozen manifest, and class-weight report match the current P1 files.

The local training source selects the best state only when validation loss decreases, then loads that best state before evaluating test. The V2 history contains only training and validation metrics; test metrics/predictions are final separate outputs. The comparison JSON also records `test_used_for_model_selection: false`. This is consistent evidence that the test set was reserved. However, the backup does not include a hash of the exact executed Kaggle script or a Kaggle execution log, so the artifacts cannot independently prove the run's execution sequence beyond those records and the reviewed training code.

## Test identity and label verification

- Frozen manifest test assignments: 344 compounds.
- V1 test predictions: 344 rows, 344 unique InChIKeys.
- V2 test predictions: 344 rows, 344 unique InChIKeys.
- V1 keys, V2 keys and frozen test keys are identical; no extra or missing test compounds.
- For HIA, HOB20 and HOB50, V1 and V2 labels and missing-value locations match each other and the cleaned P1 dataset exactly.
- Endpoint supports are consequently identical across versions: HIA 178 observed (138 positive, 40 negative); HOB20 210 (154, 56); HOB50 213 (113, 100).
- Independently recomputed V2 test metrics from its row-level predictions match the saved V2 test metrics and comparison JSON at threshold 0.5.

The recovered `baseline_validation_predictions.csv` contains 344 V1 validation rows; their keys and labels/missingness match the frozen validation assignments. The backup does **not** include row-level V2 validation probabilities. V2 validation comparisons below are therefore based on the saved aggregate `validation_metrics.json`, training history, and comparison JSON, not a new row-level recalculation or paired validation bootstrap.

## Validation performance

Threshold is 0.5. Specificity and balanced accuracy are independently checked from the saved confusion matrices. PR-AUC is average precision (AP). Missing labels are excluded endpoint-wise.

| Endpoint | Version | n (pos / neg) | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HIA | V1 | 174 (160 / 14) | 0.845 | 0.929 | 0.900 | 0.214 | 0.914 | 0.557 | 0.628 | 0.949 |
| HIA | V2 | 174 (160 / 14) | 0.799 | 0.931 | 0.844 | 0.286 | 0.885 | 0.565 | 0.583 | 0.933 |
| HOB20 | V1 | 232 (174 / 58) | 0.703 | 0.790 | 0.822 | 0.345 | 0.806 | 0.583 | 0.604 | 0.794 |
| HOB20 | V2 | 232 (174 / 58) | 0.685 | 0.789 | 0.793 | 0.362 | 0.791 | 0.578 | 0.616 | 0.791 |
| HOB50 | V1 | 236 (126 / 110) | 0.631 | 0.623 | 0.786 | 0.455 | 0.695 | 0.620 | 0.687 | 0.700 |
| HOB50 | V2 | 236 (126 / 110) | 0.636 | 0.632 | 0.762 | 0.491 | 0.691 | 0.626 | 0.680 | 0.690 |

Validation: HOB20 ROC-AUC rises 0.012 but AP falls 0.003, while F1 falls 0.015. HOB50 ROC-AUC and AP fall about 0.007 and 0.010 respectively; F1 falls slightly. V2 HIA validation ROC-AUC falls 0.045 and F1 falls 0.029. These are aggregate validation results; no claim of validation-set statistical significance is made.

## Frozen test performance

| Endpoint | Version | n (pos / neg) | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HIA | V1 | 178 (138 / 40) | 0.843 | 0.966 | 0.826 | 0.900 | 0.891 | 0.863 | 0.938 | 0.977 |
| HIA | V2 | 178 (138 / 40) | 0.843 | 0.966 | 0.826 | 0.900 | 0.891 | 0.863 | 0.952 | 0.984 |
| HOB20 | V1 | 210 (154 / 56) | 0.643 | 0.806 | 0.675 | 0.554 | 0.735 | 0.614 | 0.685 | 0.834 |
| HOB20 | V2 | 210 (154 / 56) | 0.633 | 0.808 | 0.656 | 0.571 | 0.724 | 0.614 | 0.689 | 0.827 |
| HOB50 | V1 | 213 (113 / 100) | 0.662 | 0.688 | 0.664 | 0.660 | 0.676 | 0.662 | 0.687 | 0.703 |
| HOB50 | V2 | 213 (113 / 100) | 0.648 | 0.676 | 0.646 | 0.650 | 0.661 | 0.648 | 0.698 | 0.713 |

At the fixed threshold, HIA classifications are identical; its V2 ranking metrics are higher. HOB20 has slightly higher precision/specificity and ROC-AUC, but lower recall, F1, accuracy and AP. HOB50 ROC-AUC/AP increase modestly, while accuracy, precision, recall, specificity, F1 and balanced accuracy decline. These mixed point estimates do not establish general improvement.

## Paired bootstrap uncertainty

A class-stratified paired bootstrap used 10,000 replicates, seed 20261112, and percentile 95% intervals. Within each endpoint, positive and negative observed test compounds were resampled separately, preserving endpoint class counts; the same sampled compound indices were applied to V1 and V2. Differences are V2 minus V1. Thus accuracy intervals are conditional on the observed test class proportions. All endpoints contain both classes, so ROC-AUC is calculable. No threshold was tuned.

| Endpoint | Metric | Difference | Paired bootstrap 95% CI |
|---|---|---:|---:|
| HIA | ROC-AUC | +0.0147 | [+0.0016, +0.0304] |
| HIA | F1 | +0.0000 | [-0.0156, +0.0157] |
| HIA | Accuracy | +0.0000 | [-0.0225, +0.0225] |
| HIA | Balanced accuracy | +0.0000 | [-0.0375, +0.0375] |
| HOB20 | ROC-AUC | +0.0041 | [-0.0204, +0.0284] |
| HOB20 | F1 | -0.0110 | [-0.0369, +0.0134] |
| HOB20 | Accuracy | -0.0095 | [-0.0381, +0.0190] |
| HOB20 | Balanced accuracy | -0.0008 | [-0.0349, +0.0349] |
| HOB50 | ROC-AUC | +0.0106 | [-0.0121, +0.0333] |
| HOB50 | F1 | -0.0150 | [-0.0490, +0.0171] |
| HOB50 | Accuracy | -0.0141 | [-0.0469, +0.0188] |
| HOB50 | Balanced accuracy | -0.0138 | [-0.0477, +0.0194] |

The HIA ROC-AUC interval is nominally above zero, but it is a small difference, and this exploratory analysis is not multiplicity-adjusted across endpoints and metrics. It is not evidence that descriptor fusion improves the target endpoints. For HOB20 and HOB50, all listed paired intervals include zero; there is no clear evidence of a test-set improvement in these metrics. The small endpoint-specific test subsets, including 56 HOB20 negatives and 40 HIA negatives, limit precision. Bootstrap intervals quantify resampling uncertainty under this split; they do not address external-domain generalization or dataset bias. Resampling compounds rather than scaffold clusters assumes compound-level independence; related molecules within a scaffold may make these intervals optimistic. A scaffold-cluster bootstrap would be a useful sensitivity analysis.

## HOB20/HOB50 consistency

Counts are over all 344 molecules in each split, including rows with missing endpoint labels, because both heads produce probabilities for every compound. A probability inversion means `P(HOB50) > P(HOB20)`; a binary contradiction at 0.5 means HOB50 is positive while HOB20 is negative.

| Split | Diagnostic | V1 | V2 | Change |
|---|---|---:|---:|---:|
| Validation | Probability inversions | 103 / 344 (29.9%) | 128 / 344 (37.2%) | +25 |
| Validation | Contradictory binary predictions | 1 / 344 | 0 / 344 | -1 |
| Test | Probability inversions | 79 / 344 (23.0%) | 88 / 344 (25.6%) | +9 |
| Test | Contradictory binary predictions | 3 / 344 | 0 / 344 | -3 |

Both models have zero observed HOB50-positive/HOB20-negative label violations on validation and test. V2 removes the few binary contradictions at threshold 0.5, but increases probability inversions; it does not make the outputs consistently ordered. No hierarchy constraint or post-processing was applied. Interpretation remains conditional on the endpoints measuring the same underlying bioavailability quantity at nested thresholds.

## Training curve and overfitting

V1 history has 10 epochs: training batch loss decreases from 0.377198 to 0.310269, while validation loss decreases from 0.369462 to its minimum 0.351158 at epoch 10. Validation loss does not turn upward in the recorded V1 epochs; this history shows no clear late-validation overfit signal, though the train/validation gap grows.`r`n`r`nV2 history contains seven epochs; validation loss reaches its minimum at epoch 4 and then rises in each subsequent epoch while training loss continues decreasing. The configured patience is 3, consistent with stopping after epoch 7 without a new validation-loss minimum. V2 best validation loss (0.351728) is slightly higher than V1 best validation loss (0.351158), a descriptive difference of about 0.00057.

| Epoch | Mean training batch loss | Validation loss |
|---:|---:|---:|
| 1 | 0.366305 | 0.359962 |
| 2 | 0.344502 | 0.355695 |
| 3 | 0.336878 | 0.354002 |
| 4 | 0.316668 | **0.351728** |
| 5 | 0.315568 | 0.354208 |
| 6 | 0.299344 | 0.353887 |
| 7 | 0.296310 | 0.355106 |

The widening train/validation gap after epoch 4 is a mild overfitting signal; validation loss remains near its minimum, so it is not evidence of severe collapse. Selecting epoch 4 by minimum validation masked loss is corroborated by history, checkpoint metadata, run summary, and exact checkpoint/best-model weight equality. Test metrics are stored separately and do not appear in the epoch history.

## Limitations and recommendation

1. **Production recommendation: retain V1.** V2 has no consistent HOB improvement: HOB20 and HOB50 test F1/accuracy decline, and paired intervals for both endpoints include zero for ROC-AUC, F1 and accuracy. Small ROC-AUC/AP point gains for HOB are uncertain and not corroborated by validation for HOB50.
2. V2 improves HOB binary consistency at the 0.5 cutoff but worsens probability-order consistency, so that diagnostic is not a general improvement.
3. Validation V2 aggregate metrics are available, but a row-level V2 validation probability file is absent. Therefore, validation metrics and validation consistency counts could not be independently recomputed or paired-resampled.
4. The epoch history supports validation-based selection, and both run outputs and the reviewed training code show test evaluation after selection. The backup has no executed script hash or Kaggle log; retain this as a provenance limitation.
5. Test supports are modest and endpoint-specific due to missing labels; these CIs concern only the frozen scaffold test sample. They do not demonstrate external or nutraceutical generalization.
6. The single nominal HIA ROC-AUC CI above zero is exploratory, does not apply to HOB20/HOB50, and is not multiplicity-adjusted.

The V2 artifact is worth preserving for follow-up or independent validation, but current evidence does not justify replacing the established V1 production baseline.


