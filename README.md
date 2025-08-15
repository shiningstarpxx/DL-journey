# DL-Journey: 经典深度学习网络复现项目

## 项目简介

本项目旨在复现经典深度学习网络，特别针对macOS环境进行了优化。由于macOS没有NVIDIA GPU，我们使用以下技术栈：

- **PyTorch**: 深度学习框架
- **Metal Performance Shaders (MPS)**: Apple的GPU加速框架
- **CPU**: 作为备选计算设备

## 项目架构

```
DL-journey/
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── alexnet.py         # AlexNet实现
│   ├── vgg.py             # VGG系列 (待实现)
│   ├── resnet.py          # ResNet系列 (待实现)
│   └── base_model.py      # 基础模型类
├── datasets/              # 数据集处理
│   ├── __init__.py
│   ├── imagenet.py        # ImageNet数据集
│   └── cifar.py           # CIFAR数据集
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── device.py          # 设备管理
│   ├── visualization.py   # 可视化工具
│   └── metrics.py         # 评估指标
├── configs/               # 配置文件
│   ├── alexnet_config.py  # AlexNet配置
│   └── base_config.py     # 基础配置
├── experiments/           # 实验脚本
│   ├── train_alexnet.py   # AlexNet训练
│   └── evaluate_alexnet.py # AlexNet评估
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 环境要求

- macOS 12.3+ (支持MPS)
- Python 3.8+
- PyTorch 1.12+ (支持MPS)

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd DL-journey

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练AlexNet

```bash
python experiments/train_alexnet.py --config configs/alexnet_config.py
```

### 评估AlexNet

```bash
python experiments/evaluate_alexnet.py --model_path checkpoints/alexnet_best.pth
```

## 设备支持

项目会自动检测并使用最佳可用设备：

1. **MPS (Metal Performance Shaders)**: Apple Silicon Mac的GPU加速
2. **CPU**: 所有macOS设备的备选方案

## 性能优化

- 针对macOS环境优化的数据加载
- 内存使用优化
- 批处理大小自适应调整

## 扩展性

项目设计支持轻松添加新的网络架构：

1. 在 `models/` 目录下添加新的模型文件
2. 在 `configs/` 目录下添加配置文件
3. 在 `experiments/` 目录下添加训练脚本

## 贡献

欢迎提交Issue和Pull Request来改进项目！
