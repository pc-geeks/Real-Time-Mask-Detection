# Real-Time Mask Detection

用 **YOLOv8 + ResNet-50** 做实时口罩检测：摄像头找人，再判断是否佩戴口罩。

覆盖 **目标检测、迁移学习、实时推理流水线**。

```
Webcam  →  YOLOv8n（检测 person）  →  裁剪行人框  →  ResNet-50（Mask / No Mask）  →  窗口显示
```

## 这个项目能说明什么

| 能力 | 在代码里怎么体现 |
|------|------------------|
| 两阶段视觉流水线 | 检测（YOLO）和分类（ResNet）解耦，而不是把整件事丢给一个黑盒 |
| 迁移学习 | ImageNet 预训练 ResNet-50；冻结浅层，只微调 `layer4` + 分类头 |
| 训练工程 | 数据增强、验证集划分、学习率衰减、只保存验证集最优权重 |
| 实时系统 | 摄像头循环、训练时 GPU/CPU 自动选择、关闭窗口退出 |

本机训练结果：**验证集准确率约 99.7%**（Kaggle Face Mask 数据集相对干净，准确率不是唯一看点，重点是流水线设计）。

## 技术栈

`Python` · `PyTorch` · `Torchvision` · `YOLOv8` · `OpenCV`

| 模块 | 选型 | 说明 |
|------|------|------|
| 行人检测 | YOLOv8n | COCO 预训练，约 6MB，无需再训练 |
| 口罩分类 | ResNet-50 | ImageNet 预训练后微调 10 epoch |
| 数据 | [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) | 约 7500 张，两类基本均衡 |

**微调策略：** 冻结 `conv1`～`layer3`，只训练 `layer4` 和新建的 2 类 `fc`。口罩是局部外观变化，底层边缘/纹理特征可以直接复用，这样在 CPU 上也能较快收敛。

## 快速开始

环境：Python 3.9+，摄像头。GPU 可选。

```bash
pip install -r requirements.txt
```

```bash
# 1. 配置 Kaggle API：https://www.kaggle.com/settings → Create New Token
#    把 kaggle.json 放到 ~/.kaggle/ 或 C:\Users\<你>\.kaggle\
python download_dataset.py

# 2. 微调分类器（权重保存到 models/resnet_mask.pth）
python train.py

# 3. 打开摄像头实时检测，关闭窗口即退出
python detect.py
```

## 项目结构

```
mask-detection/
├── detect.py              # 实时检测入口
├── train.py               # ResNet-50 微调
├── download_dataset.py    # 从 Kaggle 拉取并整理数据
├── requirements.txt
├── models/                # train.py 生成 resnet_mask.pth
└── dataset/               # with_mask / without_mask
```

数据集和 `.pth` 权重体积较大，不进 Git；克隆后按上面三步即可复现。

## 设计取舍

- **检测 + 分类，而不是直接训 YOLO 口罩框。** 行人检测用现成 COCO 权重，分类任务数据更好找，适合小数据和短训练时间。
- **冻住浅层。** 全量微调 2500 万参数在 CPU 上慢，也更容易在 7k 张图上过拟合。
- **验证时不做随机翻转 / 颜色抖动。** 评估要对齐真实输入分布。

## 可以继续做的方向

1. 用人脸检测或只裁行人框上半身，减少衣服、背景干扰
2. 把口罩三类化：正确佩戴 / 未佩戴 / 佩戴不规范
3. 导出 ONNX / TensorRT，提高 CPU 帧率
4. 加混淆矩阵和错误样本可视化，不只看 accuracy

## 常见问题

**`cv2.imshow` 报 `function is not implemented`**  
环境里装了 `opencv-python-headless`。卸掉它，保留 `opencv-python`。
