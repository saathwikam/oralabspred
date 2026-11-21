# HOB20/HOB50 Improvement Feasibility Study

## Scope and safeguards

This is a validation-only feasibility analysis. It loads the existing best ChemBERTa checkpoint, uses only rows assigned `validation` by the frozen split manifest for probability scoring and threshold comparisons, and does not access test predictions for any selection. The current production cutoff remains 0.5. No weights, datasets, labels, class weights, split assignments, or inference behavior were changed. Test metrics reported in earlier baseline material were not used to select thresholds or experiments.

Threshold candidates include a fixed 0.05 grid and all observed validation probability boundaries. The latter are exploratory and especially prone to overfitting a small validation subset. Precision-recall AUC is average precision (AP), a ranking metric; it does not vary with a decision threshold.

## Existing baseline and split context

- Architecture: one shared `DeepChem/ChemBERTa-77M-MLM` encoder, CLS-token pooling, dropout 0.1, and three independent sigmoid binary heads (HIA, HOB20, HOB50). HOB20 and HOB50 currently have no shared/ordinal constraint.
- Frozen split: 1,601 train, 344 validation, 344 test. The prior split audit found no InChIKey or scaffold groups crossing partitions. This study did not create or alter a split.
- Training-only positive weights recorded by P1: HOB20 0.2965; HOB50 0.8090. These are not recalculated or changed here.
- The supplied endpoint definitions must be confirmed before enforcing a hard hierarchy: HOB20 and HOB50 appear to represent nested ≥20% and ≥50% bioavailability cutoffs, but source-definition details were not independently verified in this study.

## Label coverage and nesting

Missing labels remain missing; metrics use only observed endpoint labels.

| Split | HOB20 observed | HOB20 missing | HOB50 observed | HOB50 missing | Both observed | HOB20 only | HOB50 only | Neither | Label cases HOB20=0, HOB50=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,067 | 534 | 1,089 | 512 | 1,067 | 0 | 22 | 512 | 0 |
| Validation | 232 | 112 | 236 | 108 | 232 | 0 | 4 | 108 | 0 |
| Test (descriptive only; not used for this study's choices) | 210 | 134 | 213 | 131 | 210 | 0 | 3 | 131 | 0 |

Across the 1,509 compounds with both labels observed, the label pair counts are HOB20=0/HOB50=0: 358; 0/1: 0; 1/0: 310; 1/1: 841. The absence of the 0/1 pattern is consistent with nested thresholds in these data, conditional on the endpoints truly measuring the same quantity and compatible assays/definitions. Twenty-nine HOB50 labels are observed while HOB20 is missing, so those cases cannot verify nesting.

## Validation performance and threshold tradeoffs

Validation observed support is HOB20: 174 positive / 58 negative (75.0% positive); HOB50: 126 positive / 110 negative (53.4% positive). All metrics below use validation labels only.

| Endpoint and criterion | Cutoff | F1 | Precision | Recall / sensitivity | Specificity | Balanced accuracy | Accuracy | PR-AUC (AP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HOB20 current | 0.500 | 0.806 | 0.790 | 0.822 | 0.345 | 0.583 | 0.703 | 0.794 |
| HOB20 max F1, empirical boundary | 0.301 | 0.867 | 0.769 | 0.994 | 0.103 | 0.549 | 0.772 | 0.794 |
| HOB20 max balanced accuracy, empirical boundary | 0.553 | 0.764 | 0.808 | 0.724 | 0.483 | 0.603 | 0.664 | 0.794 |
| HOB50 current | 0.500 | 0.695 | 0.623 | 0.786 | 0.455 | 0.620 | 0.631 | 0.700 |
| HOB50 max F1, empirical boundary | 0.420 | 0.719 | 0.597 | 0.905 | 0.300 | 0.602 | 0.623 | 0.700 |
| HOB50 max balanced accuracy, empirical boundary | 0.566 | 0.661 | 0.681 | 0.643 | 0.655 | 0.649 | 0.648 | 0.700 |

Cutoffs are selected separately per endpoint. “Max” means an in-sample optimum on this one validation split, not a recommended deployment setting. For HOB20, maximizing F1 predicts nearly every positive and leaves specificity at 0.103; this illustrates the effect of its 75% positive prevalence. Its max-balanced-accuracy candidate improves specificity from 0.345 to 0.483 but lowers recall from 0.822 to 0.724. For HOB50, the max-F1 candidate raises recall but lowers precision and specificity; the balanced-accuracy candidate improves specificity and balanced accuracy at the cost of recall and F1.

ROC-AUC/AP do not change when only the cutoff changes: validation HOB20 ROC-AUC 0.604, AP 0.794; HOB50 ROC-AUC 0.687, AP 0.700. Thus threshold adjustment changes the operating point, not ranking/discrimination. HOB20 AP is only modestly above the 0.75 positive prevalence baseline; HOB50 AP is above its 0.534 prevalence baseline. These estimates are based on modest endpoint-specific validation samples and have no confidence intervals here.

The full 0.10–0.90 threshold grid, including F1, precision, recall, specificity, balanced accuracy, accuracy, ROC-AUC and AP, is in `hob_validation_threshold_metrics.csv`. Actual validation probabilities and their masked labels are in `hob_validation_predictions.csv`.

## Independent-head consistency

At the existing 0.5 cutoff, validation contains 1 compound predicted HOB50-positive and HOB20-negative. Independently, HOB50 probability exceeds HOB20 probability for 103/344 validation molecules (29.9%). Among the 232 with both labels observed, the labels have no nested-threshold violation, but the model outputs are not constrained to respect that relationship. The probability ordering diagnostic includes rows where either label is missing; it is descriptive only.

## Candidate improvement approaches

### 1. RDKit descriptor late-fusion baseline

Concatenate a compact, predeclared set of physicochemical/descriptive RDKit features with the shared ChemBERTa representation before endpoint heads, or compare a small descriptor-only regularized classifier as a diagnostic control. Molecular size, lipophilicity, polarity, hydrogen-bonding capacity, and flexibility can encode absorption-relevant properties that a small endpoint dataset may not learn reliably from fine-tuning alone. This is feasible with the existing RDKit dependency and keeps the pretrained encoder. With only 2,289 compounds and scaffolded splits, use a small fixed descriptor set, fit imputers/scalers on training rows only, keep endpoint masks, and avoid feature selection against the test set. Treat it as an experiment, not an expected improvement.

### 2. Hierarchy-aware HOB20/HOB50 multitask objective

If authoritative endpoint documentation confirms common, nested bioavailability thresholds, compare the independent-head baseline with a monotonic/ordinal formulation that enforces or penalizes HOB50 probability exceeding HOB20 probability. The observed paired labels are fully nested (0 such label violations among 1,509 pairs), and the current independent heads show a measurable probability-order inconsistency. Preserve missing-label masks, including the 29 HOB50-only observed cases. This is scientifically motivated for consistency, but could constrain predictions incorrectly if assay definitions differ; validate the definition first. A soft penalty is a lower-risk first comparison than hard post-processing because it does not silently rewrite outputs.

## Recommendation and selection policy

**Recommended first experiment: RDKit descriptor late fusion**, with the existing ChemBERTa model and frozen split as baseline. It directly tests whether explicit, interpretable molecular properties add signal to the weak HOB20 ranking and moderate HOB50 ranking without presuming a stricter endpoint relationship. Keep the candidate feature set and training recipe small and fixed. Select using validation-only, predeclared metrics (ROC-AUC/AP for ranking plus balanced accuracy/F1 for a chosen operating objective); do not tune repeatedly on the test set. After locking a candidate, evaluate the frozen test set once for final comparison.

The hierarchy-aware experiment is the second priority, conditional on authoritative confirmation of endpoint nesting. Threshold calibration can be evaluated as a separate operating-point decision, but it cannot improve ROC-AUC/AP and should not be confused with model improvement. Keep the production cutoff at 0.5 until a later approved calibration decision and external validation.

## Reproducibility artifacts

- `run_hob_threshold_feasibility.py`: validation-only scorer and threshold analysis; never scores the test split.
- `hob_validation_predictions.csv`: validation predictions and labels, retaining missing labels as NaN.
- `hob_validation_threshold_metrics.csv`: fixed-grid validation metrics.
- `hob_validation_feasibility.json`: validation support, candidate cutoffs, and HOB nesting diagnostics.

