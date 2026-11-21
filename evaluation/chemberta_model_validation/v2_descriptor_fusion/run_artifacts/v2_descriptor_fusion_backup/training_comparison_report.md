# V2 Descriptor-Fusion Training Comparison

Best checkpoint selected at epoch 4 by validation masked loss (0.351728).
Test data was evaluated only after this selection. Threshold: 0.5 for both model versions.
Metrics use observed endpoint labels only; PR-AUC is average precision.

| Split | Endpoint | Version | N | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | HIA | V1 | 174 | 0.845 | 0.929 | 0.900 | 0.214 | 0.914 | 0.557 | 0.628 | 0.949 |
| validation | HIA | V2 | 174 | 0.799 | 0.931 | 0.844 | 0.286 | 0.885 | 0.565 | 0.583 | 0.933 |
| validation | HOB20 | V1 | 232 | 0.703 | 0.790 | 0.822 | 0.345 | 0.806 | 0.583 | 0.604 | 0.794 |
| validation | HOB20 | V2 | 232 | 0.685 | 0.789 | 0.793 | 0.362 | 0.791 | 0.578 | 0.616 | 0.791 |
| validation | HOB50 | V1 | 236 | 0.631 | 0.623 | 0.786 | 0.455 | 0.695 | 0.620 | 0.687 | 0.700 |
| validation | HOB50 | V2 | 236 | 0.636 | 0.632 | 0.762 | 0.491 | 0.691 | 0.626 | 0.680 | 0.690 |
| test | HIA | V1 | 178 | 0.843 | 0.966 | 0.826 | 0.900 | 0.891 | 0.863 | 0.938 | 0.977 |
| test | HIA | V2 | 178 | 0.843 | 0.966 | 0.826 | 0.900 | 0.891 | 0.863 | 0.952 | 0.984 |
| test | HOB20 | V1 | 210 | 0.643 | 0.806 | 0.675 | 0.554 | 0.735 | 0.614 | 0.685 | 0.834 |
| test | HOB20 | V2 | 210 | 0.633 | 0.808 | 0.656 | 0.571 | 0.724 | 0.614 | 0.689 | 0.827 |
| test | HOB50 | V1 | 213 | 0.662 | 0.688 | 0.664 | 0.660 | 0.676 | 0.662 | 0.687 | 0.703 |
| test | HOB50 | V2 | 213 | 0.648 | 0.676 | 0.646 | 0.650 | 0.661 | 0.648 | 0.698 | 0.713 |

## HOB20/HOB50 consistency

The JSON artifact records V1 and V2 probability-order and binary-threshold violations on the same rows. No hierarchy constraint or post-processing is applied.

## Run configuration

- Seed: 42
- Optimizer: AdamW; learning rate 2e-05; batch size 16
- Epoch cap / patience: 10 / 3; max length 256
- Environment: {'python': '3.13.16', 'torch': '2.11.0+cu130', 'transformers': '5.20.2', 'rdkit': '2026.09.2', 'numpy': '2.1.3', 'pandas': '2.3.3', 'sklearn': '1.6.1', 'device': 'cuda', 'gpu': 'Tesla T4', 'gpu_memory_available_gib_at_start': 14.44415283203125, 'gpu_memory_total_gib': 14.56219482421875, 'cuda_version': '13.0', 'seed': 42, 'mixed_precision_enabled': True}
