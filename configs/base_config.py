"""
基础配置类
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseConfig:
    """基础配置类"""
    
    # 数据配置
    data_dir: str = './data'
    dataset_name: str = 'cifar10'
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    augment: bool = True
    
    # 模型配置
    model_name: str = 'alexnet'
    num_classes: int = 10
    pretrained: bool = False
    modern: bool = True
    
    # 训练配置
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    
    # 设备配置
    device: str = 'auto'  # 'auto', 'cpu', 'mps', 'cuda'
    
    # 保存配置
    save_dir: str = './checkpoints'
    save_freq: int = 10
    log_dir: str = './logs'
    
    # 其他配置
    seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_model_config(self):
        """获取模型配置"""
        return {
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'modern': self.modern
        }
    
    def get_data_config(self):
        """获取数据配置"""
        return {
            'data_dir': self.data_dir,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'image_size': self.image_size,
            'augment': self.augment
        }
    
    def get_training_config(self):
        """获取训练配置"""
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_gamma': self.scheduler_gamma
        }
