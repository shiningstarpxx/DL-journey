#!/bin/bash

# DL-Journey 项目安装脚本

echo "🚀 开始安装 DL-Journey 项目依赖..."

# 检查Python版本
echo "📋 检查Python版本..."
python3 --version

# 检查pip
echo "📦 检查pip..."
python3 -m pip --version

# 升级pip
echo "⬆️ 升级pip..."
python3 -m pip install --upgrade pip

# 安装PyTorch (针对macOS优化)
echo "🔥 安装PyTorch (支持MPS)..."
python3 -m pip install torch torchvision torchaudio

# 安装其他依赖
echo "📚 安装其他依赖包..."
python3 -m pip install -r requirements.txt

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p data checkpoints logs

echo "✅ 安装完成！"
echo ""
echo "🎯 下一步："
echo "1. 运行测试: python3 test_project.py"
echo "2. 开始训练: python3 experiments/train_alexnet.py"
echo "3. 查看TensorBoard: tensorboard --logdir logs"
echo ""
echo "📖 更多信息请查看 README.md"
