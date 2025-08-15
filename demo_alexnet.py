"""
AlexNet快速演示脚本
展示模型训练和评估的完整流程
"""

import os
import sys
import torch
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_alexnet():
    """AlexNet演示函数"""
    
    # 导入项目模块
    from models.alexnet import create_alexnet
    from datasets.cifar import CIFAR10Dataset
    from configs.alexnet_config import AlexNetConfig
    from utils.device import get_best_device, optimize_for_device
    from utils.metrics import calculate_accuracy
    
    logger.info("🚀 开始AlexNet演示...")
    
    # 获取设备
    device = get_best_device()
    logger.info(f"使用设备: {device}")
    
    # 创建配置
    config = AlexNetConfig()
    config.epochs = 2  # 只训练2个epoch用于演示
    config.batch_size = 16
    config.num_workers = 0  # 避免多进程问题
    
    # 创建模型
    logger.info("📦 创建AlexNet模型...")
    model_config = config.get_alexnet_config()
    model = create_alexnet(**model_config)
    model = optimize_for_device(model, device)
    
    # 打印模型信息
    logger.info(f"模型参数数量: {model.count_parameters():,}")
    logger.info(f"模型大小: {model.get_model_size_mb():.2f} MB")
    
    # 创建数据集
    logger.info("📊 加载CIFAR-10数据集...")
    data_config = config.get_data_config()
    data_config['num_workers'] = 0
    dataset = CIFAR10Dataset(**data_config)
    train_loader, test_loader = dataset.get_dataloaders()
    
    # 设置训练组件
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    logger.info("🎯 开始训练...")
    model.train()
    
    for epoch in range(config.epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_correct += calculate_accuracy(output, target) * target.size(0)
            total_samples += target.size(0)
            
            # 更新进度条
            if batch_idx % 100 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{calculate_accuracy(output, target):.2%}'
                })
        
        # 计算epoch统计
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / total_samples
        
        logger.info(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2%}')
    
    # 评估模型
    logger.info("🔍 评估模型...")
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_correct += calculate_accuracy(output, target) * target.size(0)
            total_samples += target.size(0)
    
    test_accuracy = total_correct / total_samples
    logger.info(f"测试准确率: {test_accuracy:.2%}")
    
    # 单张图像预测演示
    logger.info("🖼️ 单张图像预测演示...")
    model.eval()
    
    # 获取一张测试图像
    for data, target in test_loader:
        sample_image = data[0:1].to(device)
        sample_target = target[0]
        break
    
    with torch.no_grad():
        output = model(sample_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = dataset.get_class_names()
    predicted_class_name = class_names[predicted_class]
    true_class_name = class_names[sample_target.item()]
    
    logger.info(f"预测类别: {predicted_class_name} (置信度: {confidence:.2%})")
    logger.info(f"真实类别: {true_class_name}")
    
    logger.info("✅ AlexNet演示完成！")
    
    return {
        'device': str(device),
        'model_params': model.count_parameters(),
        'model_size_mb': model.get_model_size_mb(),
        'test_accuracy': test_accuracy,
        'predicted_class': predicted_class_name,
        'true_class': true_class_name,
        'confidence': confidence
    }

if __name__ == '__main__':
    # 确保在虚拟环境中运行
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"MPS可用: {torch.backends.mps.is_available()}")
        
        results = demo_alexnet()
        
        print("\n" + "="*50)
        print("演示结果摘要:")
        print("="*50)
        for key, value in results.items():
            print(f"{key}: {value}")
        print("="*50)
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已激活虚拟环境并安装了所有依赖")
        print("运行: source venv/bin/activate && pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
