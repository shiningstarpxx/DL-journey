"""
LeNet专用配置
"""

from .base_config import BaseConfig

class LeNetConfig(BaseConfig):
    """LeNet专用配置类"""
    
    def __init__(self):
        super().__init__()
        
        # LeNet特定配置
        self.model_name = 'lenet'
        self.num_classes = 10  # CIFAR-10
        self.dataset_name = 'cifar10'
        
        # LeNet训练配置
        self.epochs = 100
        self.learning_rate = 0.001
        self.batch_size = 64  # LeNet较小，可以使用更大的批处理
        self.weight_decay = 1e-4
        
        # LeNet模型配置
        self.pretrained = False
        self.modern = True  # 使用现代版本
        self.dropout_rate = 0.5
        self.input_channels = 3  # 彩色图像
        
        # 针对macOS的优化配置
        self.num_workers = 2  # macOS上减少工作进程数
        self.image_size = 224
        
        # 保存配置
        self.save_dir = './checkpoints/lenet'
        self.log_dir = './logs/lenet'
    
    def get_lenet_config(self):
        """获取LeNet特定配置"""
        return {
            'dropout_rate': self.dropout_rate,
            'input_channels': self.input_channels,
            **self.get_model_config()
        }
