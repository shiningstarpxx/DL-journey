"""
项目测试脚本
验证所有组件是否正常工作
"""

import os
import sys
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_device_utils():
    """测试设备管理模块"""
    logger.info("测试设备管理模块...")
    
    try:
        from utils.device import get_best_device, get_device_info, print_device_info
        
        device = get_best_device()
        info = get_device_info()
        
        logger.info(f"最佳设备: {device}")
        logger.info(f"设备信息: {info}")
        
        print_device_info()
        logger.info("✓ 设备管理模块测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 设备管理模块测试失败: {e}")
        return False

def test_models():
    """测试模型模块"""
    logger.info("测试模型模块...")
    
    try:
        from models.alexnet import AlexNet, AlexNetModern, create_alexnet
        
        # 测试原始AlexNet
        model = AlexNet(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        logger.info(f"原始AlexNet输出形状: {output.shape}")
        
        # 测试现代版AlexNet
        model_modern = AlexNetModern(num_classes=10)
        output_modern = model_modern(x)
        logger.info(f"现代版AlexNet输出形状: {output_modern.shape}")
        
        # 测试工厂函数
        model_factory = create_alexnet(num_classes=10, modern=True)
        output_factory = model_factory(x)
        logger.info(f"工厂函数创建模型输出形状: {output_factory.shape}")
        
        logger.info("✓ 模型模块测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 模型模块测试失败: {e}")
        return False

def test_datasets():
    """测试数据集模块"""
    logger.info("测试数据集模块...")
    
    try:
        from datasets.cifar import CIFAR10Dataset
        
        # 创建小批次数据集进行测试
        dataset = CIFAR10Dataset(batch_size=2, num_workers=0)
        train_loader, test_loader = dataset.get_dataloaders()
        
        # 获取一个批次的数据
        for data, target in train_loader:
            logger.info(f"CIFAR-10数据形状: {data.shape}")
            logger.info(f"CIFAR-10标签形状: {target.shape}")
            logger.info(f"CIFAR-10类别名称: {dataset.get_class_names()}")
            break
        
        logger.info("✓ 数据集模块测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 数据集模块测试失败: {e}")
        return False

def test_metrics():
    """测试评估指标模块"""
    logger.info("测试评估指标模块...")
    
    try:
        from utils.metrics import calculate_accuracy, calculate_top_k_accuracy
        
        # 创建模拟数据
        output = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        
        # 测试准确率计算
        accuracy = calculate_accuracy(output, target)
        logger.info(f"计算准确率: {accuracy:.4f}")
        
        # 测试Top-K准确率
        top5_accuracy = calculate_top_k_accuracy(output, target, k=5)
        logger.info(f"Top-5准确率: {top5_accuracy:.4f}")
        
        logger.info("✓ 评估指标模块测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 评估指标模块测试失败: {e}")
        return False

def test_configs():
    """测试配置模块"""
    logger.info("测试配置模块...")
    
    try:
        from configs.alexnet_config import AlexNetConfig
        
        config = AlexNetConfig()
        
        logger.info(f"模型配置: {config.get_model_config()}")
        logger.info(f"数据配置: {config.get_data_config()}")
        logger.info(f"训练配置: {config.get_training_config()}")
        
        logger.info("✓ 配置模块测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 配置模块测试失败: {e}")
        return False

def test_end_to_end():
    """端到端测试"""
    logger.info("测试端到端流程...")
    
    try:
        from models.alexnet import create_alexnet
        from datasets.cifar import CIFAR10Dataset
        from utils.device import get_best_device, optimize_for_device
        from utils.metrics import calculate_accuracy
        
        # 获取设备
        device = get_best_device()
        
        # 创建模型
        model = create_alexnet(num_classes=10, modern=True)
        model = optimize_for_device(model, device)
        
        # 创建数据集
        dataset = CIFAR10Dataset(batch_size=2, num_workers=0)
        train_loader, _ = dataset.get_dataloaders()
        
        # 前向传播测试
        model.eval()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                output = model(data)
                accuracy = calculate_accuracy(output, target)
            
            logger.info(f"端到端测试 - 输出形状: {output.shape}, 准确率: {accuracy:.4f}")
            break
        
        logger.info("✓ 端到端测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 端到端测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始项目测试...")
    
    tests = [
        test_device_utils,
        test_models,
        test_datasets,
        test_metrics,
        test_configs,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！项目可以正常使用。")
        return True
    else:
        logger.error("❌ 部分测试失败，请检查错误信息。")
        return False

if __name__ == '__main__':
    main()
