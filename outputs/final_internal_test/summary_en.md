# Final four-model evaluation

The SIIM–ISIC 2020 data were split by patient. The internal test set contains 4,763 images from 309 patients, including 82 melanoma images and 4,681 benign images. The first-round loss-function experiments also used this internal test set.

The primary model averages the scores of two padding-augmented ResNet-50 models (seeds 42 and 43) and an image-plus-metadata branch built from the seed-42 image features. The main comparator averages the two image-only scores. The metadata branch uses age, sex, and lesion site, encoded using the training-set configuration. Each model uses its own threshold selected on the validation set before the test evaluation. Scores at or above the threshold are classified as positive.

| Model | Split | Fixed threshold¹ | Melanomas detected | Missed melanomas | False positives | Sensitivity | Specificity | Precision | Accuracy | AP | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dual ResNet + metadata (primary) | Validation | 0.01359096 | 76 | 8 | 885 | 90.48% | 80.49% | 7.91% | 80.67% | 0.1858 | 0.9051 |
| Dual ResNet + metadata (primary) | Internal test | 0.01359096 | 61 | 21 | 877 | 74.39% | 81.26% | 6.50% | 81.15% | 0.2846 | 0.8806 |
| Padding-augmented ResNet | Validation | 0.01156694 | 76 | 8 | 943 | 90.48% | 79.21% | 7.46% | 79.42% | 0.1703 | 0.8972 |
| Padding-augmented ResNet | Internal test | 0.01156694 | 64 | 18 | 939 | 78.05% | 79.94% | 6.38% | 79.91% | 0.2669 | 0.8762 |
| Image-only dual ResNet | Validation | 0.01284592 | 76 | 8 | 879 | 90.48% | 80.62% | 7.96% | 80.80% | 0.1603 | 0.8996 |
| Image-only dual ResNet | Internal test | 0.01284592 | 62 | 20 | 877 | 75.61% | 81.26% | 6.60% | 81.17% | 0.2780 | 0.8765 |
| Single ResNet + metadata | Validation | 0.00924276 | 76 | 8 | 1175 | 90.48% | 74.10% | 6.08% | 74.39% | 0.1949 | 0.9047 |
| Single ResNet + metadata | Internal test | 0.00924276 | 69 | 13 | 1179 | 84.15% | 74.81% | 5.53% | 74.97% | 0.2767 | 0.8827 |

¹ Thresholds are shown to eight decimal places. The evaluation used the [full-precision frozen protocol](frozen_protocol_en.json).

## Findings

- On validation, all four models detected 76 of 84 melanoma images. The primary model produced 58 fewer false positives than the padding-augmented single ResNet (885 vs. 943) and six more than the image-only dual ResNet (885 vs. 879).
- On the internal test set, the primary model detected 61 of 82 melanoma images. It had the highest average precision (AP, 0.2846) of the four models and tied the image-only dual ResNet at 877 false positives. The image-only dual ResNet detected one more melanoma.
- All four models fell short of 90% sensitivity at their fixed thresholds on the internal test set. Interpret false positives alongside missed melanomas because test sensitivity varied across models.

[Metrics CSV](comparison_en.csv) · [Original Chinese report](中文测试报告.md)
