"""
AlexNet专用配置
"""

from .base_config import BaseConfig

class AlexNetConfig(BaseConfig):
    """AlexNet专用配置类"""
    
    def __init__(self):
        super().__init__()
        
        # AlexNet特定配置
        self.model_name = 'alexnet'
        self.num_classes = 10  # CIFAR-10
        self.dataset_name = 'cifar10'
        
        # AlexNet训练配置
        self.epochs = 100
        self.learning_rate = 0.001
        self.batch_size = 32
        self.weight_decay = 1e-4
        
        # AlexNet模型配置
        self.pretrained = False
        self.modern = True  # 使用现代版本
        self.dropout_rate = 0.5
        
        # 针对macOS的优化配置
        self.num_workers = 2  # macOS上减少工作进程数
        self.image_size = 224
        
        # 保存配置
        self.save_dir = './checkpoints/alexnet'
        self.log_dir = './logs/alexnet'
    
    def get_alexnet_config(self):
        """获取AlexNet特定配置"""
        return {
            'dropout_rate': self.dropout_rate,
            **self.get_model_config()
        }
