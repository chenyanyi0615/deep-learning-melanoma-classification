# 实验输出汇总

## 周一汇报材料

| 环节 | 材料 |
| --- | --- |
| 问题与数据 | [ISIC 2020 患者级划分及类别统计](reports/dataset.json) |
| 基线与阈值 | [三种损失及验证阈值对照](threshold_analysis/comparison.md)、[第一轮固定阈值测试](test_evaluation/summary_zh.md) |
| 图像模型改进 | [稳定微调](round2/中文实验报告.md)、[填充增强](padding_augmentation/中文实验报告.md)、[第二个随机种子](padding_seed43_repeat/中文实验报告.md) |
| 元数据与最终融合 | [元数据分支对照](metadata_fusion/中文实验报告.md)、[最终四模型报告](final_internal_test/中文测试报告.md)、[完整指标表](final_internal_test/comparison.csv)、[固定模型与阈值](final_internal_test/frozen_protocol.json) |
| 现场 demo | [运行说明](../melanoma_demo/README.md)、[一页导入图](../melanoma_demo/Fusion-intro.pdf) |
| 问答备份 | [CBAM layer4](cbam_control/中文实验报告.md)、[CBAM layer3](cbam_layer3_control/中文实验报告.md)、[困难负样本挖掘](hnm_control/中文实验报告.md)、[2019 补充数据](supplement2019_control/中文对照实验报告.md)、[EfficientNet 集成](efficientnet_ensemble/中文实验报告.md) |

验证集上，融合模型与填充增强单模型各检出 76/84 张黑色素瘤，误报分别为 885 和 943 张。按预先固定的阈值，融合模型在内部测试集检出 61/82 张，误报 877 张。纯双 ResNet 检出 62/82 张，误报同为 877 张。该内部测试集也用于第一轮损失函数评估。

## 其他材料

- [错误分析](error_analysis/stable_bce_224_seed42/中文错误分析报告.md)
- [填充检查](padding_sensitivity/中文检查报告.md)

本目录收录汇总报告、指标和配置。部分早期训练状态为上传时的快照。
