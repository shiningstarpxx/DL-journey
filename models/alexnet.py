"""
AlexNet模型实现
基于2012年ImageNet竞赛的经典网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class AlexNet(BaseModel):
    """
    AlexNet模型实现
    
    原始论文: "ImageNet Classification with Deep Convolutional Neural Networks"
    作者: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
    年份: 2012
    """
    
    def __init__(self, num_classes=1000, pretrained=False, dropout_rate=0.5):
        """
        初始化AlexNet
        
        Args:
            num_classes (int): 分类数量
            pretrained (bool): 是否使用预训练权重
            dropout_rate (float): Dropout比率
        """
        self.dropout_rate = dropout_rate
        super(AlexNet, self).__init__(num_classes, pretrained)
    
    def _build_model(self):
        """
        构建AlexNet架构
        """
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四个卷积块
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第五个卷积块
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, 3, 224, 224]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """
        获取特征表示（不包含分类层）
        
        Args:
            x: 输入张量
            
        Returns:
            特征张量
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        # 返回最后一个全连接层之前的特征
        for i, layer in enumerate(self.classifier):
            if i == len(self.classifier) - 1:  # 最后一层（分类层）
                break
            x = layer(x)
        return x
    
    def _load_pretrained_weights(self):
        """
        加载预训练权重
        """
        try:
            # 尝试加载torchvision的预训练权重
            import torchvision.models as models
            pretrained_model = models.alexnet(pretrained=True)
            
            # 复制权重
            self.load_state_dict(pretrained_model.state_dict(), strict=False)
            logger.info("成功加载torchvision预训练的AlexNet权重")
        except Exception as e:
            logger.warning(f"无法加载预训练权重: {e}")
            logger.info("使用随机初始化的权重")

class AlexNetModern(BaseModel):
    """
    现代版本的AlexNet
    使用现代深度学习最佳实践进行改进
    """
    
    def __init__(self, num_classes=1000, pretrained=False, dropout_rate=0.5):
        """
        初始化现代版AlexNet
        
        Args:
            num_classes (int): 分类数量
            pretrained (bool): 是否使用预训练权重
            dropout_rate (float): Dropout比率
        """
        self.dropout_rate = dropout_rate
        super(AlexNetModern, self).__init__(num_classes, pretrained)
    
    def _build_model(self):
        """
        构建现代版AlexNet架构
        """
        # 特征提取层（使用现代最佳实践）
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # 第四个卷积块
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # 第五个卷积块
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # 自适应池化
            nn.Flatten(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        使用现代方法初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, 3, 224, 224]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """
        获取特征表示（不包含分类层）
        
        Args:
            x: 输入张量
            
        Returns:
            特征张量
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        # 返回最后一个全连接层之前的特征
        for i, layer in enumerate(self.classifier):
            if i == len(self.classifier) - 1:  # 最后一层（分类层）
                break
            x = layer(x)
        return x

def create_alexnet(num_classes=1000, pretrained=False, modern=True, dropout_rate=0.5):
    """
    创建AlexNet模型的工厂函数
    
    Args:
        num_classes (int): 分类数量
        pretrained (bool): 是否使用预训练权重
        modern (bool): 是否使用现代版本
        dropout_rate (float): Dropout比率
        
    Returns:
        AlexNet或AlexNetModern实例
    """
    if modern:
        return AlexNetModern(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        return AlexNet(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)

if __name__ == "__main__":
    # 测试模型
    device = torch.device("cpu")
    
    # 测试原始AlexNet
    print("=== 原始AlexNet ===")
    model = AlexNet(num_classes=1000)
    model.print_model_info()
    
    # 测试输入
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试现代版AlexNet
    print("\n=== 现代版AlexNet ===")
    model_modern = AlexNetModern(num_classes=1000)
    model_modern.print_model_info()
    
    output_modern = model_modern(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_modern.shape}")
