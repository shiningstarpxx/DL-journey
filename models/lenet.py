"""
LeNet模型实现
基于1998年Yann LeCun提出的经典卷积神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class LeNet(BaseModel):
    """
    LeNet-5模型实现
    
    原始论文: "Gradient-Based Learning Applied to Document Recognition"
    作者: Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner
    年份: 1998
    """
    
    def __init__(self, num_classes=10, pretrained=False, dropout_rate=0.5, input_channels=1):
        """
        初始化LeNet
        
        Args:
            num_classes (int): 分类数量
            pretrained (bool): 是否使用预训练权重
            dropout_rate (float): Dropout比率
            input_channels (int): 输入通道数 (1为灰度，3为彩色)
        """
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels
        super(LeNet, self).__init__(num_classes, pretrained)
    
    def _build_model(self):
        """
        构建LeNet-5架构
        """
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(self.input_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积层
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
        )
        
        # 自适应池化层，适应不同输入尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(84, self.num_classes),
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, height, width]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
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
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        # 返回最后一个全连接层之前的特征
        for i, layer in enumerate(self.classifier):
            if i == len(self.classifier) - 1:  # 最后一层（分类层）
                break
            x = layer(x)
        return x

class LeNetModern(BaseModel):
    """
    现代版本的LeNet
    使用现代深度学习最佳实践进行改进，支持彩色图像
    """
    
    def __init__(self, num_classes=10, pretrained=False, dropout_rate=0.5, input_channels=3):
        """
        初始化现代版LeNet
        
        Args:
            num_classes (int): 分类数量
            pretrained (bool): 是否使用预训练权重
            dropout_rate (float): Dropout比率
            input_channels (int): 输入通道数 (1为灰度，3为彩色)
        """
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels
        super(LeNetModern, self).__init__(num_classes, pretrained)
    
    def _build_model(self):
        """
        构建现代版LeNet架构
        """
        # 特征提取层（使用现代最佳实践）
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 自适应池化层，适应不同输入尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(128, self.num_classes),
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
            x: 输入张量 [batch_size, channels, height, width]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
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
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        # 返回最后一个全连接层之前的特征
        for i, layer in enumerate(self.classifier):
            if i == len(self.classifier) - 1:  # 最后一层（分类层）
                break
            x = layer(x)
        return x

def create_lenet(num_classes=10, pretrained=False, modern=True, input_channels=3):
    """
    创建LeNet模型的工厂函数
    
    Args:
        num_classes (int): 分类数量
        pretrained (bool): 是否使用预训练权重
        modern (bool): 是否使用现代版本
        input_channels (int): 输入通道数
        
    Returns:
        LeNet或LeNetModern实例
    """
    if modern:
        return LeNetModern(num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)
    else:
        return LeNet(num_classes=num_classes, pretrained=pretrained, input_channels=input_channels)

if __name__ == "__main__":
    # 测试模型
    device = torch.device("cpu")
    
    # 测试原始LeNet (灰度图像)
    print("=== 原始LeNet (灰度图像) ===")
    model = LeNet(num_classes=10, input_channels=1)
    x = torch.randn(1, 1, 28, 28)  # MNIST尺寸
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    model.print_model_info()
    
    # 测试原始LeNet (彩色图像)
    print("\n=== 原始LeNet (彩色图像) ===")
    model_color = LeNet(num_classes=10, input_channels=3)
    x_color = torch.randn(1, 3, 32, 32)  # CIFAR-10尺寸
    output_color = model_color(x_color)
    print(f"输入形状: {x_color.shape}")
    print(f"输出形状: {output_color.shape}")
    model_color.print_model_info()
    
    # 测试现代版LeNet (彩色图像)
    print("\n=== 现代版LeNet (彩色图像) ===")
    model_modern = LeNetModern(num_classes=10, input_channels=3)
    x_modern = torch.randn(1, 3, 224, 224)  # CIFAR-10尺寸
    output_modern = model_modern(x_modern)
    print(f"输入形状: {x_modern.shape}")
    print(f"输出形状: {output_modern.shape}")
    model_modern.print_model_info()
