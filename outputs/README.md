# Presentation materials

| Presentation section | Materials |
| --- | --- |
| Problem and data | [SIIM–ISIC 2020 patient-level split and class counts](reports/dataset.json) |
| Baselines, model development, and Q&A | [Threshold selection, padding augmentation, metadata, and exploratory experiments](development_en.md) |
| Final model and results | [Four-model validation and internal test report](final_internal_test/summary_en.md), [metrics CSV](final_internal_test/comparison_en.csv), [frozen evaluation protocol](final_internal_test/frozen_protocol_en.json) |
| Live demo | [Run guide and case sequence](../melanoma_demo/README.md), [one-slide introduction (PDF)](../melanoma_demo/Fusion-intro.pdf) |

The final fusion model detected **76/84** melanoma images on validation with **885** false positives. At the same fixed threshold on the internal test set, it detected **61/82** with **877** false positives. The image-only dual ResNet detected **62/82**, also with **877** false positives. This was a re-evaluation on the internal test set used in the first round.

[Original Chinese index](README.zh.md)
