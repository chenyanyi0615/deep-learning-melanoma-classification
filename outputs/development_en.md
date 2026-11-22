# Development experiments and Q&A notes

The project compares specificity and false-positive counts at a validation sensitivity of at least 90%. All rows in the two tables below use validation thresholds that detected 76 of 84 melanoma images (90.48%). The [final report](final_internal_test/summary_en.md) presents the internal test comparison at fixed thresholds.

## Baseline and image preprocessing

At the default threshold of 0.5, plain BCE and Focal Loss each detected 0 of 84 melanoma images on validation. Weighted BCE detected 77 of 84. At the selected validation thresholds, BCE, Weighted BCE, and Focal Loss each detected 76 images and produced 1,464, 1,436, and 1,842 false positives, respectively. With the same thresholds on the internal test set, their sensitivities were 82.93%, 78.05%, and 85.37%. [Validation threshold comparison](threshold_analysis/comparison.md) · [Original baseline test report in Chinese](test_evaluation/summary_zh.md)

The stable fine-tuning experiment trained the classification head first, then fine-tuned the final ResNet block and the head while keeping batch normalization frozen. Several training and checkpoint-selection choices changed between the first two rounds, so the improvement reflects the combined changes. The padding experiment then varied only the padding color during training: fixed gray with 75% probability and the image’s mean color with 25% probability. Validation and test images use fixed gray padding. Images are resized while preserving aspect ratio and padded to 224 × 224.

| Model | False positives | Specificity | AUROC |
| --- | ---: | ---: | ---: |
| Weighted BCE baseline | 1,436 | 68.34% | 0.8624 |
| Stable fine-tuning with BCE, 224 px, seed 42 | 1,041 | 77.05% | 0.8956 |
| Padding-augmented ResNet, seed 42 | 943 | 79.21% | 0.8972 |
| Padding-augmented ResNet, seed 43 | 1,040 | 77.07% | 0.8932 |
| Image-only dual ResNet | 879 | 80.62% | 0.8996 |
| Dual ResNet + metadata (primary) | 885 | 80.49% | 0.9051 |

[Stable fine-tuning report](round2/中文实验报告.md) · [Padding augmentation report](padding_augmentation/中文实验报告.md) · [Seed-43 repeat](padding_seed43_repeat/中文实验报告.md)

## What metadata added

The image-only and image-plus-metadata heads use the same frozen seed-42 ResNet image features. Adding age, sex, and lesion site raised AP from 0.1815 to 0.1949 and AUROC from 0.9004 to 0.9047. At 90.48% validation sensitivity, false positives increased from 1,109 to 1,175. The final ensemble combined the metadata branch with the two original image models. Compared with the image-only dual ResNet on validation, it had six additional false positives and increased AP from 0.1603 to 0.1858. [Original metadata experiment report](metadata_fusion/中文实验报告.md) · [Final evaluation](final_internal_test/summary_en.md)

## Exploratory experiments

| Experiment | Matched control FP | Experiment FP | Observation |
| --- | ---: | ---: | --- |
| CBAM after ResNet layer 4 | 943 | 1,012 | False positives increased by 69. |
| CBAM after ResNet layer 3 | 943 | 1,302 | False positives increased by 359. |
| Hard negative mining | 943 | 1,407 | False positives increased by 464. |
| Add 416 verified ISIC 2019 images | 1,041 | 1,128 | False positives increased by 87. |
| Image-only ResNet head + EfficientNet-B0, 1:1 | 1,109 | 1,028 | False positives decreased by 81. |
| ResNet metadata head + EfficientNet-B0, 1:1 | 1,175 | 1,091 | False positives decreased by 84. |

The lowest false-positive count across the EfficientNet ensembles was 1,028. The metadata-head ensemble with a 2:1 ResNet-to-EfficientNet weight reached an AUROC of 0.9067 with 1,109 false positives.

[CBAM layer 4](cbam_control/中文实验报告.md) · [CBAM layer 3](cbam_layer3_control/中文实验报告.md) · [Hard negative mining](hnm_control/中文实验报告.md) · [ISIC 2019 supplement](supplement2019_control/中文对照实验报告.md) · [EfficientNet](efficientnet_ensemble/中文实验报告.md)
