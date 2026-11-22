# Deep Learning-Based Melanoma Classification

生物医学工程小组项目：SIIM-ISIC 2020、PyTorch、ResNet-50。

[周一汇报材料](outputs/README.zh.md) · [四模型最终结果](outputs/final_internal_test/中文测试报告.md) · [现场 demo](melanoma_demo/README.zh.md)

## 当前进度

完成环境检查、官方训练元数据下载、官方重复图像清单下载，以及患者级划分程序。
**官方 JPEG 已下载、CRC 校验并解压；32,701 张保留图像均已完整解码验证。**
ResNet-50 训练代码已实现。基线实际运行状态见 `outputs/experiments/bce_seed42/status.json`，
完整验证集跑完一轮后才会产生该轮指标。三种损失实验、验证阈值比较及固定方案测试集评估现已完成。

## 项目结构

```text
data/raw/train.csv          官方 v2 训练元数据（保留原文件）
data/raw/duplicates.csv     官方重复图像对
data/raw/jpeg/train/        解压后的训练 JPEG；支持子目录递归查找
data/splits/                train.csv、val.csv、test.csv 固定划分清单
scripts/prepare_data.py     数据审计、去重、患者分层划分
src/dataset.py             图像加载、预处理与训练增强
src/model.py               ResNet-50 单 logit 二分类模型
src/losses.py              BCE、Weighted BCE、Focal Loss
src/metrics.py             AUROC、敏感度、特异度、precision、F1、混淆矩阵
src/train.py               训练、验证、保存检查点及绘图
data/pretrained/           官方 ImageNet 预训练权重缓存
outputs/experiments/       每次实验的独立目录（配置、日志、权重、验证指标、图）
outputs/reports/           数据审计、环境记录；后续实验结果
outputs/figures/           后续类别分布、训练曲线、ROC、混淆矩阵
outputs/checkpoints/       后续模型权重和训练状态
requirements.txt           本机已验证可导入的版本
```

## 第一步：获取与检查数据

官方入口：https://challenge.isic-archive.com/data/ （选择 2020）。
JPEG 训练集约 23 GB、33,126 张；下载解压还需要额外磁盘空间。
官方训练标签包括 image_name、patient_id、target；v2 还包含 lesion_id。
target=1 是 melanoma，target=0 是非 melanoma；报告中可称良性类，但不应把
diagnosis=unknown 当作缺失标签或擅自删除。其他元数据的缺失会被统计；
当前图像分类阶段不使用性别、年龄等字段作模型输入。

下载训练 JPEG 并解压到 data/raw/jpeg/train/，或使用 --images 指定实际目录。
不需要下载官方无标签测试图像：本项目的测试集来自有标签训练数据的内部留出。
官方数据采用 CC-BY-NC 4.0；报告需注明来源：
International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset.
https://doi.org/10.34970/2020-ds01

在项目根目录运行（当前已安装依赖，无须重复安装）：

```bash
python3 scripts/prepare_data.py
# 图像下载并解压后，检查所有文件存在且可解码，再更新清单：
python3 scripts/prepare_data.py --images data/raw/jpeg/train --verify-images
```

元数据模式允许图像缺失，清单中的 image_path 此时为空，报告 training_ready=false。
--verify-images 模式对缺失、损坏文件报错，不能悄悄跳过。
更换文件目录后需要重新运行，清单保存绝对图像路径。

去重策略：先删除官方重复清单每对的第二张图，不使用标签挑选图像。
这会改变样本总数；原文件保持完整，报告保存删去数量和源文件 SHA256。
仅处理官方重复清单，不宣称发现了所有视觉近重复。

划分策略：患者的任一图像为阳性，即该患者属于阳性分层；固定 seed=42，
先分 70%/30%，再将 30% 对半分为验证/测试。**70/15/15 是患者数比例**，
四舍五入会有细小偏差；每人图像数不同，图像比例与阳性率不强求完全相同。
脚本检查患者交集为零、患者覆盖完整、三个集合均包含两类。
缺失 patient_id 会直接报错，不能用随意填充掩盖泄漏风险。

## 第二步：训练 BCE 基线

本机在沙盒外检测到 MPS GPU，内存 8 GiB；沙盒内的 MPS 检测可能返回不可用。
默认 batch size=8、224×224、AdamW、学习率 1e-4、weight decay=1e-4、
最多 10 轮，连续 3 轮验证 AUROC 未提升则早停。微调全部网络参数。
使用 `IMAGENET1K_V2` 权重；最后全连接层随机初始化，输出单个 logit。

图像先用 JPEG 解码器缩小到至少约 448 的目标尺度，转 RGB，保持长宽比缩放并补边
到 224×224，采用 ImageNet 均值/标准差归一化。训练集额外使用水平/垂直翻转、
±15° 旋转、0.1 幅度的亮度/对比度/饱和度增强。验证集只使用固定预处理。

在项目根目录运行；已有实验目录需要使用续训参数，避免覆盖结果：

```bash
python3 -u -m src.train --device mps
# 意外退出后，从上一完整 epoch 的模型和优化器状态续训：
python3 -u -m src.train --device mps --resume
```

若第一轮还没有完成，则尚无 `last.pt` 可续训；此时需要用新的实验目录重启。
固定划分和种子，按 epoch 重新设种子；MPS 上不保证跨运行逐位相同。
训练器使用文件锁，防止同一实验被同时启动两次。

每个实验目录包含：

- `config.json`：超参数、预处理、依赖版本和划分 SHA256。
- `train.log`：后台启动时重定向保存的训练输出。
- `status.json`：当前阶段、epoch、batch、吞吐量及状态更新时间。
- `history.csv`：每个已完成 epoch 的训练损失及完整验证集指标。
- `checkpoints/best_weights.pt`：验证 AUROC 最高的模型权重。
- `checkpoints/last.pt`：上一完整 epoch 的模型、优化器和训练历史。
- `best_validation_metrics.json` / `best_validation_predictions.csv`：最佳验证指标及逐图概率。
- `figures/`：训练结束后生成损失曲线、验证 ROC 与混淆矩阵。

分类阈值固定 0.5；未预测出阳性时 precision/F1 按 0 记录。
当前训练器只评估验证集，测试集仅作划分完整性检查；最终测试在方案冻结后单独进行。
Weighted BCE/Focal Loss 接口已实现，但以下比较实验尚未启动：

```bash
python3 -u -m src.train --device mps --loss weighted_bce --output outputs/experiments/weighted_bce_seed42
python3 -u -m src.train --device mps --loss focal --output outputs/experiments/focal_seed42
```

Weighted BCE 的正类权重只由训练集计算：22903/415 ≈ 55.19。
Focal 默认 gamma=2、alpha=0.25（阳性系数，阴性系数为 0.75），不是已经调优的参数。
后续如调优，只使用验证集。三种实验保持相同划分、初始种子与训练预算。

## 完整实验流程与待办

1. 图像完整性检查与类别分布图，确认固定划分。
2. Dataset：RGB、224×224；训练集随机翻转、适度旋转/颜色增强；验证与测试
   使用固定变换。使用 ImageNet mean/std，检查预处理不会过度裁掉病灶。
3. ResNet-50 ImageNet 初始化，最后全连接层输出一个 logit。
   BCE 基线采用 BCEWithLogitsLoss；loss 前不加 sigmoid。
   预训练权重首次使用可能需要联网下载。
4. 相同划分、初始化种子、训练预算下比较普通 BCE、Weighted BCE、Focal Loss。
   pos_weight 只从训练集算 N_negative/N_positive；Focal gamma 初始可设 2，
   alpha 与其他超参数只能用验证集选择。不默认同时叠加过采样与加权。
5. 验证 AUROC 选择 checkpoint；分类阈值预先固定 0.5，若调阈值只用验证集。
   冻结三种方案后统一评估内部测试集，不能用测试结果反复选模型。
6. 保存 AUROC、sensitivity=TP/(TP+FN)、specificity=TN/(TN+FP)、precision、
   F1、混淆矩阵（行是真实类别、列是预测类别，0/1 顺序），以及逐图概率。
   AUROC 使用连续概率；零分母需明确处理。准确率不能代替这些指标。
7. 保存最佳权重、配置、种子、依赖版本、划分哈希、每轮日志、训练曲线与 ROC。
   如估计置信区间，按患者 bootstrap，保留同一患者内的图像相关性。

这里的内部测试不能代表外部医院的泛化验证，项目结果用于课程研究。

## 验证集阈值比较（已执行）

运行 `python3 -m scripts.compare_thresholds`，使用三个最佳 checkpoint 已保存的验证集分数，
分别选择满足敏感度 >=90% 的最高阈值；判定规则是 probability >= threshold。
该步骤没有重新训练、没有重新推理、没有使用测试集预测。

结果保存在 `outputs/threshold_analysis/`：`comparison.md`、`comparison.csv`、
`report.json`、各模型的完整阈值扫描 CSV、`roc_operating_points.png` 和
`confusion_matrices.png`。报告同时记录默认阈值 0.5 与所选阈值下的全部指标。

三个模型在所选阈值下均为 TP=76、FN=8（敏感度 90.48%）。BCE、Weighted BCE、
Focal Loss 的误报 FP 分别为 1464、1436、1842；Weighted BCE 比 BCE 仅少 28 个误报。
不能根据这次验证集上的小差异宣称稳定或显著优于 BCE。
阈值在同一验证集上选择和报告，指标可能乐观；最终测试敏感度不保证达到 90%。
具体阈值的完整精度、预测文件哈希和验证集哈希已保存于 report.json，供冻结后的测试评估使用。

## 固定方案测试集评估（已执行）

程序：`python3 -u -m scripts.evaluate_test --device mps`。
结果：`outputs/test_evaluation/`；中文说明见 `summary_zh.md`，完整数据见 `report.json`
和 `comparison.csv`。原始逐图概率、ROC 和六张混淆矩阵均保存在该目录。

共评估 309 位患者的 4,763 张图像（82 张黑色素瘤、4,681 张良性）。
运行前保存 `frozen_protocol.json`，记录模型权重、患者划分的 SHA256，以及来自验证集的
完整精度阈值。测试过程中没有训练、调参或重新选阈值。

验证集首选为 Weighted BCE，固定阈值 0.5129998922348022 在测试集得到敏感度 78.05%、
特异度 69.56%。BCE 和 Focal Loss 的固定阈值测试敏感度分别为 82.93%、85.37%。
三个模型都未在测试集维持 >=90% 敏感度目标。不能据此宣称任一损失函数稳定优于其他方法。
测试集现已使用；未来根据这些结果改进模型时，应明确属于进一步探索，
新的独立确认需要另一份未使用的数据或合适的嵌套验证设计。

测试评估意外中断时，可使用 `--resume` 复用完整保存的预测；该参数检查原冻结方案，
不会搜索测试阈值。已完成的评估无需再次运行。

## 第二轮自主改进（已完成 5 组实验，未超过 12 小时预算）

方案与限时见 `outputs/round2/实验方案.md`、`plan.json`；队列状态见 `outputs/round2/status.json`。
由 `scripts/round2_experiments.py` 顺序执行稳定微调 BCE、适度阳性重采样 BCE、阳性系数更高的 Focal，
再视剩余时间做最佳配置的不同种子重复及可选 320 分辨率实验。
冻结 BatchNorm 和早期骨干，先训练分类头，再微调 layer4。主指标是 >=90% 验证敏感度下的特异度。
本轮不引入数据、不使用付费服务、不重新使用已经评估过的测试集调参。
队列自动记录完整轮次的真实结果，结束时生成 `outputs/round2/中文实验报告.md`、对比 CSV 和验证曲线。
该目录还包含训练前错误分析、输入图检查及 MPS 梯度检查记录。

PyTorch 参考：
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
