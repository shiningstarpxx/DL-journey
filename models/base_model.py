"""
基础模型类
为所有深度学习模型提供通用接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """
    基础模型类，所有模型都应该继承此类
    """
    
    def __init__(self, num_classes=1000, pretrained=False):
        """
        初始化基础模型
        
        Args:
            num_classes (int): 分类数量
            pretrained (bool): 是否使用预训练权重
        """
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_name = self.__class__.__name__
        
        # 初始化模型
        self._build_model()
        
        if pretrained:
            self._load_pretrained_weights()
    
    @abstractmethod
    def _build_model(self):
        """
        构建模型架构
        子类必须实现此方法
        """
        pass
    
    def _load_pretrained_weights(self):
        """
        加载预训练权重
        子类可以重写此方法
        """
        logger.info(f"加载 {self.model_name} 预训练权重")
        # 默认实现为空，子类可以重写
    
    def forward(self, x):
        """
        前向传播
        子类必须实现此方法
        """
        raise NotImplementedError
    
    def get_features(self, x):
        """
        获取特征表示（不包含分类层）
        子类可以重写此方法
        
        Args:
            x: 输入张量
            
        Returns:
            特征张量
        """
        return self.forward(x)
    
    def count_parameters(self):
        """
        计算模型参数数量
        
        Returns:
            int: 参数总数
        """
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        """
        计算可训练参数数量
        
        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """
        获取模型大小（MB）
        
        Returns:
            float: 模型大小（MB）
        """
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def freeze_backbone(self):
        """
        冻结主干网络参数
        子类可以重写此方法
        """
        logger.info(f"冻结 {self.model_name} 主干网络参数")
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """
        解冻主干网络参数
        子类可以重写此方法
        """
        logger.info(f"解冻 {self.model_name} 主干网络参数")
        for param in self.parameters():
            param.requires_grad = True
    
    def get_optimizer(self, lr=0.001, weight_decay=1e-4):
        """
        获取优化器
        
        Args:
            lr (float): 学习率
            weight_decay (float): 权重衰减
            
        Returns:
            torch.optim.Optimizer: 优化器
        """
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer, num_epochs, steps_per_epoch):
        """
        获取学习率调度器
        
        Args:
            optimizer: 优化器
            num_epochs (int): 总训练轮数
            steps_per_epoch (int): 每轮步数
            
        Returns:
            torch.optim.lr_scheduler._LRScheduler: 学习率调度器
        """
        total_steps = num_epochs * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path (str): 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path (str): 模型路径
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 {path} 加载")
    
    def print_model_info(self):
        """
        打印模型信息
        """
        print(f"模型名称: {self.model_name}")
        print(f"分类数量: {self.num_classes}")
        print(f"预训练: {self.pretrained}")
        print(f"参数总数: {self.count_parameters():,}")
        print(f"可训练参数: {self.count_trainable_parameters():,}")
        print(f"模型大小: {self.get_model_size_mb():.2f} MB")
        
        # 打印模型结构
        print("\n模型结构:")
        print(self)
