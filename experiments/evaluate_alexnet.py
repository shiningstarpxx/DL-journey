"""
AlexNet评估脚本
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alexnet import create_alexnet
from datasets.cifar import CIFAR10Dataset
from configs.alexnet_config import AlexNetConfig
from utils.device import get_best_device, optimize_for_device
from utils.metrics import calculate_metrics_summary, print_metrics_summary

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlexNetEvaluator:
    """AlexNet评估器"""
    
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        self.device = get_best_device()
        
        # 初始化模型和数据
        self._setup_model()
        self._setup_data()
        
        logger.info(f"评估设备: {self.device}")
    
    def _setup_model(self):
        """设置模型"""
        logger.info("加载AlexNet模型...")
        
        # 创建模型
        model_config = self.config.get_alexnet_config()
        self.model = create_alexnet(**model_config)
        
        # 加载预训练权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"成功加载模型权重: {self.model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"成功加载模型权重: {self.model_path}")
        else:
            logger.warning(f"模型文件不存在: {self.model_path}")
            logger.info("使用随机初始化的模型")
        
        # 优化设备
        self.model = optimize_for_device(self.model, self.device)
        self.model.eval()
        
        # 打印模型信息
        self.model.print_model_info()
    
    def _setup_data(self):
        """设置数据"""
        logger.info("加载测试数据集...")
        
        # 创建数据集（不使用数据增强）
        data_config = self.config.get_data_config()
        data_config['augment'] = False  # 评估时不使用数据增强
        
        self.dataset = CIFAR10Dataset(**data_config)
        _, self.test_loader = self.dataset.get_dataloaders()
        
        logger.info(f"测试集批次数: {len(self.test_loader)}")
    
    def evaluate(self):
        """评估模型"""
        logger.info("开始评估AlexNet...")
        
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())
        
        # 合并所有输出和目标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算评估指标
        class_names = self.dataset.get_class_names()
        summary = calculate_metrics_summary(
            all_outputs, all_targets, 
            self.config.num_classes, 
            class_names
        )
        
        # 打印评估结果
        print_metrics_summary(summary)
        
        # 保存评估结果
        self._save_evaluation_results(summary)
        
        return summary
    
    def _save_evaluation_results(self, summary):
        """保存评估结果"""
        import json
        import numpy as np
        
        # 准备保存的数据
        results = {
            'model_path': self.model_path,
            'device': str(self.device),
            'accuracy': summary['accuracy'],
            'top5_accuracy': summary['top5_accuracy'],
            'precision': summary['precision'],
            'recall': summary['recall'],
            'f1': summary['f1'],
            'class_accuracy': summary['class_accuracy'],
            'confusion_matrix': summary['confusion_matrix'].tolist()
        }
        
        # 保存为JSON文件
        results_path = os.path.join(self.config.log_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {results_path}")
    
    def predict_single_image(self, image_path):
        """预测单张图像"""
        from PIL import Image
        import torchvision.transforms as transforms
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 获取类别名称
        class_names = self.dataset.get_class_names()
        predicted_class_name = class_names[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估AlexNet')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--image_path', type=str, default=None, help='单张图像路径（用于测试）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = AlexNetConfig()
    
    # 创建评估器
    evaluator = AlexNetEvaluator(config, args.model_path)
    
    # 评估模型
    summary = evaluator.evaluate()
    
    # 如果提供了单张图像，进行预测
    if args.image_path and os.path.exists(args.image_path):
        logger.info(f"预测单张图像: {args.image_path}")
        result = evaluator.predict_single_image(args.image_path)
        
        print(f"\n预测结果:")
        print(f"预测类别: {result['predicted_class_name']} (ID: {result['predicted_class']})")
        print(f"置信度: {result['confidence']:.4f}")
        
        # 显示前5个预测结果
        class_names = evaluator.dataset.get_class_names()
        probabilities = result['probabilities']
        top5_indices = probabilities.argsort()[-5:][::-1]
        
        print(f"\n前5个预测结果:")
        for i, idx in enumerate(top5_indices):
            print(f"  {i+1}. {class_names[idx]}: {probabilities[idx]:.4f}")

if __name__ == '__main__':
    main()
