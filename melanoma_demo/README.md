# 黑色素瘤筛查现场演示

单页界面用真实皮肤镜图像和年龄、性别、病灶部位现场计算三个分支的分数。最终融合分数是两个填充增强 ResNet 和图像＋元数据分支分数的平均值。推理将图像等比例缩放并使用固定填充补至 224 × 224。模型分数达到验证集预先选定的阈值 `0.0135909608307573` 时，界面标记为阳性。

- [演示导入页（PDF）](Fusion-intro.pdf)
- [最终验证集与内部测试集结果](../outputs/final_internal_test/中文测试报告.md)
- [固定阈值及模型配置](../outputs/final_internal_test/frozen_protocol.json)

## 在演示电脑运行

使用组内完整的 `Melanoma-Fusion-Demo.zip`。完整解压后，目录下应有 `assets/model.pt`、`assets/config.json` 和三张案例图像。建议用 64 位 Python 3.12。首次启动需要联网安装推理依赖，安装完成后可离线运行，CPU 即可。

- Windows：双击 `Start-Windows.bat`。
- macOS 或 Linux：在解压目录执行 `python3 launch.py`。
- 已安装 PyTorch、NumPy、Pillow 时，可以直接执行 `python demo.py`。

启动后保持终端打开，并在浏览器访问终端显示的本机地址。公开目录提供界面和推理源码，现场运行所需的模型文件和案例素材来自组内演示包。

## 现场流程

每次先点 **Run fusion** 执行推理，再点 **Reveal label** 揭示数据集标签。用 **Compare models** 展开四个方案各自的分数和固定阈值。

- **Case A** 展示正确检出的黑色素瘤，解释三个分数如何取平均。
- **Case B** 展示一张良性图像。三个对照均判阳性，融合模型正确判阴性。这个案例用于解释一次具体误报如何避免。
- **Case C** 展示漏检。真实标签为黑色素瘤，融合分数略低于固定阈值。
