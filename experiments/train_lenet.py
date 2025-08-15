"""
LeNet训练脚本
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import create_lenet
from datasets.cifar import CIFAR10Dataset
from configs.lenet_config import LeNetConfig
from utils.device import get_best_device, optimize_for_device
from utils.metrics import calculate_accuracy, calculate_loss

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeNetTrainer:
    """LeNet训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = get_best_device()
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 初始化模型、数据、优化器等
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_logging()
        
        logger.info(f"训练设备: {self.device}")
        logger.info(f"模型参数数量: {self.model.count_parameters():,}")
    
    def _setup_model(self):
        """设置模型"""
        logger.info("初始化LeNet模型...")
        
        # 创建模型
        model_config = self.config.get_lenet_config()
        self.model = create_lenet(**model_config)
        
        # 优化设备
        self.model = optimize_for_device(self.model, self.device)
        
        # 打印模型信息
        if self.config.verbose:
            self.model.print_model_info()
    
    def _setup_data(self):
        """设置数据"""
        logger.info("加载数据集...")
        
        # 创建数据集
        data_config = self.config.get_data_config()
        self.dataset = CIFAR10Dataset(**data_config)
        
        # 获取数据加载器
        self.train_loader, self.test_loader = self.dataset.get_dataloaders()
        
        logger.info(f"训练集批次数: {len(self.train_loader)}")
        logger.info(f"测试集批次数: {len(self.test_loader)}")
    
    def _setup_optimizer(self):
        """设置优化器"""
        training_config = self.config.get_training_config()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=training_config['scheduler_step_size'],
            gamma=training_config['scheduler_gamma']
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def _setup_logging(self):
        """设置日志记录"""
        self.writer = SummaryWriter(self.config.log_dir)
        
        # 记录配置信息
        self.writer.add_text('Config', str(self.config.__dict__))
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_correct += calculate_accuracy(output, target)
            total_samples += target.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{calculate_accuracy(output, target):.2%}'
            })
            
            # 记录到tensorboard
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', calculate_accuracy(output, target), step)
        
        # 计算epoch统计
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples
        
        logger.info(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.2%}')
        
        return avg_loss, avg_accuracy
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                total_correct += calculate_accuracy(output, target)
                total_samples += target.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = total_correct / total_samples
        
        logger.info(f'Validation: Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2%}')
        
        # 记录到tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', avg_accuracy, epoch)
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, epoch, val_accuracy, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_accuracy,
            'config': self.config.__dict__
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.save_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f'保存最佳模型，验证准确率: {val_accuracy:.2%}')
        
        # 定期保存
        if (epoch + 1) % self.config.save_freq == 0:
            periodic_path = os.path.join(self.config.save_dir, f'epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
    
    def train(self):
        """训练模型"""
        logger.info("开始训练LeNet...")
        
        best_accuracy = 0.0
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_accuracy = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_accuracy = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存检查点
            is_best = val_accuracy > best_accuracy
            if is_best:
                best_accuracy = val_accuracy
            
            self.save_checkpoint(epoch, val_accuracy, is_best)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            logger.info(f'Epoch {epoch+1} 完成 - LR: {current_lr:.6f}')
        
        logger.info(f"训练完成！最佳验证准确率: {best_accuracy:.2%}")
        self.writer.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练LeNet')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--device', type=str, default=None, help='设备类型')
    
    args = parser.parse_args()
    
    # 加载配置
    config = LeNetConfig()
    
    # 覆盖配置参数
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device
    
    # 创建训练器并开始训练
    trainer = LeNetTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
