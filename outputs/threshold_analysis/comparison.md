# Validation threshold comparison

**Validation only. No new training and no test-set evaluation.**

Select each checkpoint's highest threshold with sensitivity >= 90%.

| Model | Policy | Threshold | AUROC | Sensitivity | Specificity | Precision | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BCE | default_0.5 | 0.5 | 0.8607 | 0.00% | 100.00% | 0.00% | 0.0000 | 0 | 84 |
| BCE | sensitivity_target | 0.0040058484 | 0.8607 | 90.48% | 67.72% | 4.94% | 0.0936 | 1464 | 8 |
| Weighted BCE | default_0.5 | 0.5 | 0.8624 | 91.67% | 66.27% | 4.79% | 0.0911 | 1530 | 7 |
| Weighted BCE | sensitivity_target | 0.51299989 | 0.8624 | 90.48% | 68.34% | 5.03% | 0.0952 | 1436 | 8 |
| Focal Loss | default_0.5 | 0.5 | 0.8442 | 0.00% | 100.00% | 0.00% | 0.0000 | 0 | 84 |
| Focal Loss | sensitivity_target | 0.079363614 | 0.8442 | 90.48% | 59.39% | 3.96% | 0.0759 | 1842 | 8 |

These thresholds are now recorded for subsequent frozen test evaluation.
Validation-selected results are optimistic and do not demonstrate test-set generalization.
ROC and confusion-matrix figures are in this directory.
