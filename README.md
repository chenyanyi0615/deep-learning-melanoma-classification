# Deep Learning-Based Melanoma Classification

A biomedical engineering group project using SIIM–ISIC 2020, PyTorch, and ResNet-50. The project explores how to detect more melanoma images while controlling false positives.

[Presentation materials](outputs/README.md) · [Final results](outputs/final_internal_test/summary_en.md) · [Live demo](melanoma_demo/README.md) · [Original Chinese project guide](README.zh.md)

## Data and evaluation

After removing 425 duplicate rows, the dataset contained 32,701 images from 2,056 patients, including 581 melanoma images. Patients were assigned to a single split.

| Split | Patients | Images | Melanoma images |
| --- | ---: | ---: | ---: |
| Training | 1,439 | 23,318 | 415 |
| Validation | 308 | 4,620 | 84 |
| Internal test | 309 | 4,763 | 82 |

Model development compared specificity and false positives at a validation sensitivity of at least 90%. The final model averages predictions from two independently trained, padding-augmented ResNet-50 models and an image-plus-metadata branch that uses age, sex, and lesion site. The internal test comparison used the model-specific thresholds fixed on validation.

## Repository layout

- [Image preprocessing](src/dataset.py) and [ResNet-50 implementation](src/model.py).
- [Metadata branch training](scripts/train_metadata_fusion.py) and [final evaluation script](scripts/evaluate_final_models.py).
- [Aggregate experiment reports](outputs/README.md) and [demo source and run guide](melanoma_demo/README.md).

Use the team's full demo archive for live inference. The original project guide contains the detailed environment and data preparation instructions.
