# Cell Image Segmentation

基于 PyTorch 的细胞图像语义分割项目，使用 U-Net 对灰度显微细胞图像进行前景/背景二分类分割。项目包含数据读取与增强、U-Net 模型定义、训练脚本、实验报告，以及从报告中整理出的关键结果图片。

![Qualitative segmentation results](docs/images/qualitative-results.png)

## 项目亮点

- 使用经典 U-Net 编码器-解码器结构，包含 skip connection、transpose convolution、batch normalization、ReLU 和 dropout。
- 针对灰度细胞图像实现 `Dataset`，自动匹配 `scans/` 与 `labels/` 中的图像-mask 对。
- 数据增强包含水平/垂直翻转、缩放、旋转、gamma correction 和 elastic transform。
- 在 dataloader 中加入 histogram equalization，用于改善低灰度细胞图像导致的空 mask 预测问题。
- 训练过程使用 Weights & Biases 记录 loss、accuracy 等指标。

## 项目结构

```text
Cell-image-segmentation/
├── README.md
├── .gitignore
├── Description.pdf              # 作业/任务描述
├── report.pdf                   # 实验报告 PDF
├── report.docx                  # 实验报告源文件
├── docs/
│   └── images/                  # README 展示用结果图
│       ├── accuracy-comparison.png
│       ├── epoch-loss-comparison.png
│       ├── qualitative-results.png
│       └── sample-cell.jpeg
└── materials/
    ├── assignment_2.ipynb       # Notebook 版本实验
    ├── README.txt               # 原始简要说明
    ├── data/
    │   └── cells/
    │       ├── scans/           # 输入细胞图像
    │       └── labels/          # 对应分割标签
    └── src/
        ├── dataloader.py        # 数据集读取、预处理与增强
        ├── model.py             # U-Net 模型
        ├── train.py             # 训练、验证和可视化脚本
        ├── requirements.txt     # Python 依赖
        └── run.sh               # Linux/macOS shell 启动脚本
```

> 当前仓库中还包含本地虚拟环境、W&B 运行日志、`__pycache__` 和 checkpoint 等训练产物；新的 `.gitignore` 已将这些内容加入忽略规则，后续提交时建议不再纳入版本管理。

## 环境配置

建议使用 Python 3.9+。在 Windows PowerShell 中可以按下面方式创建独立环境：

```powershell
cd materials/src
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS：

```bash
cd materials/src
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要使用 CUDA，请根据本机 CUDA 版本安装匹配的 PyTorch wheel。当前 `requirements.txt` 中使用的是 `torch==2.5.1+cu118`、`torchvision==0.20.1+cu118`。

## 运行训练

训练脚本依赖相对路径，推荐从 `materials/src` 目录启动：

```bash
cd materials/src
python train.py
```

默认训练配置位于 [materials/src/train.py](materials/src/train.py)：

- image size: `572`
- epochs: `25`
- batch size: `4` for DataLoader, loss logging 中按 `batch_size=2` 归一化
- optimizer: 前期 Adam，后期 SGD
- scheduler: `StepLR(step_size=10, gamma=0.9)`
- loss: `CrossEntropyLoss`
- checkpoint: `materials/src/checkpoint.pt`

如需关闭 W&B，可在运行前设置离线模式：

```bash
wandb offline
python train.py
```

## 数据说明

数据位于 `materials/data/cells/`：

- `scans/`: 38 张灰度细胞扫描图
- `labels/`: 38 张对应分割标签

`Cell_data` 默认使用 `train_test_split=0.8` 划分训练集和测试集，并对训练集启用随机增强。

![Sample cell image](docs/images/sample-cell.jpeg)

## 模型结构

模型定义在 [materials/src/model.py](materials/src/model.py)，整体为标准 U-Net：

- Contracting path: 4 个 down step，每个 step 包含两层卷积和 max pooling。
- Bottleneck: 1024 通道 two-conv block。
- Expansive path: 4 个 up step，使用 transpose convolution 上采样，并与 encoder 特征做 center crop 后拼接。
- Output: `1x1` convolution 输出 2 类 logits。

## 实验结果

报告中的最佳实验设置为 random augmentation + histogram equalization，并加入 Adam/SGD、StepLR、dropout 等改动。根据 `report.docx` 的记录，最终 accuracy 约为 **0.85663**，相比未加入 histogram equalization 的多组实验有明显提升。

![Accuracy comparison](docs/images/accuracy-comparison.png)

![Epoch loss comparison](docs/images/epoch-loss-comparison.png)

更多实验细节可查看 [report.pdf](report.pdf)。

## 后续可改进方向

- 将训练参数迁移到 `argparse` 或 YAML 配置文件，便于复现实验。
- 增加 Dice score / IoU 等分割指标，而不只统计像素级 accuracy。
- 增加固定随机种子和模型评估脚本，降低实验复现成本。
- 修正 `run.sh` 中 `new_env` 与 `env` 目录名不一致的问题。
