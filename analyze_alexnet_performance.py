"""
分析AlexNet表现差的原因
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'PingFang SC', 'Hiragino Sans GB']
        plt.rcParams['axes.unicode_minus'] = False
        
        font_list = font_manager.findSystemFonts()
        chinese_fonts = []
        for font in font_list:
            try:
                font_name = font_manager.FontProperties(fname=font).get_name().lower()
                if any(name in font_name for name in ['simhei', 'arial unicode', 'dejavu', 'pingfang', 'hiragino']):
                    chinese_fonts.append(font)
            except:
                continue
        
        if chinese_fonts:
            font_path = chinese_fonts[0]
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"✅ 已设置中文字体: {font_prop.get_name()}")
        else:
            print("⚠️ 未找到中文字体，图表中的中文可能无法正常显示")
            plt.rcParams['font.family'] = 'sans-serif'
            
    except Exception as e:
        print(f"⚠️ 设置中文字体时出错: {e}")
        print("图表中的中文可能无法正常显示")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_alexnet_issues():
    """分析AlexNet的问题"""
    
    # 导入模型
    from models.alexnet import AlexNet, AlexNetModern
    from datasets.cifar import CIFAR10Dataset
    from utils.device import optimize_for_device
    from utils.metrics import calculate_accuracy
    
    device = torch.device("cpu")
    
    # 创建数据集
    dataset = CIFAR10Dataset(batch_size=32, num_workers=0, image_size=224)
    train_loader, test_loader = dataset.get_dataloaders()
    
    # 创建模型
    alexnet = AlexNet(num_classes=10)
    alexnet_modern = AlexNetModern(num_classes=10)
    
    alexnet = optimize_for_device(alexnet, device)
    alexnet_modern = optimize_for_device(alexnet_modern, device)
    
    print("="*60)
    print("AlexNet性能分析")
    print("="*60)
    
    # 1. 分析模型结构
    print("\n1. 模型结构分析:")
    print(f"AlexNet参数数量: {alexnet.count_parameters():,}")
    print(f"Modern-AlexNet参数数量: {alexnet_modern.count_parameters():,}")
    
    # 2. 分析特征图尺寸
    print("\n2. 特征图尺寸分析:")
    alexnet.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        data, _ = next(iter(train_loader))
        data = data.to(device)
        
        print(f"输入尺寸: {data.shape}")
        
        # 分析每一层的输出尺寸
        x = data
        for i, layer in enumerate(alexnet.features):
            x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                print(f"第{i+1}层 ({type(layer).__name__}): {x.shape}")
        
        # 分析全连接层
        x = torch.flatten(x, 1)
        print(f"展平后: {x.shape}")
        
        for i, layer in enumerate(alexnet.classifier):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                print(f"全连接层{i+1}: {x.shape}")
    
    # 3. 分析权重分布
    print("\n3. 权重分布分析:")
    def analyze_weight_distribution(model, model_name):
        weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights.extend(param.data.cpu().numpy().flatten())
        
        weights = np.array(weights)
        print(f"{model_name}:")
        print(f"  权重均值: {weights.mean():.6f}")
        print(f"  权重标准差: {weights.std():.6f}")
        print(f"  权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
        print(f"  零权重比例: {(weights == 0).mean():.2%}")
        
        return weights
    
    alexnet_weights = analyze_weight_distribution(alexnet, "AlexNet")
    modern_weights = analyze_weight_distribution(alexnet_modern, "Modern-AlexNet")
    
    # 4. 分析梯度流动
    print("\n4. 梯度流动分析:")
    def analyze_gradients(model, model_name):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                gradients.append((name, grad_norm))
        
        print(f"{model_name}梯度范数:")
        for name, grad_norm in sorted(gradients, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {name}: {grad_norm:.6f}")
        
        return gradients
    
    alexnet_grads = analyze_gradients(alexnet, "AlexNet")
    modern_grads = analyze_gradients(alexnet_modern, "Modern-AlexNet")
    
    # 5. 分析激活函数输出
    print("\n5. 激活函数输出分析:")
    def analyze_activations(model, model_name):
        model.eval()
        with torch.no_grad():
            data, _ = next(iter(train_loader))
            data = data.to(device)
            
            activations = []
            x = data
            
            for i, layer in enumerate(model.features):
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activation_stats = {
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'dead_ratio': (x == 0).float().mean().item()
                    }
                    activations.append((f"ReLU_{i+1}", activation_stats))
            
            print(f"{model_name}激活统计:")
            for name, stats in activations:
                print(f"  {name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 死神经元比例={stats['dead_ratio']:.2%}")
            
            return activations
    
    alexnet_acts = analyze_activations(alexnet, "AlexNet")
    modern_acts = analyze_activations(alexnet_modern, "Modern-AlexNet")
    
    # 6. 绘制分析图表
    print("\n6. 生成分析图表...")
    setup_chinese_font()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 权重分布对比
    ax1.hist(alexnet_weights, bins=50, alpha=0.7, label='AlexNet', density=True)
    ax1.hist(modern_weights, bins=50, alpha=0.7, label='Modern-AlexNet', density=True)
    ax1.set_title('权重分布对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('权重值', fontsize=12)
    ax1.set_ylabel('密度', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 梯度范数对比
    alexnet_grad_names = [name for name, _ in alexnet_grads[:10]]
    alexnet_grad_values = [value for _, value in alexnet_grads[:10]]
    modern_grad_names = [name for name, _ in modern_grads[:10]]
    modern_grad_values = [value for _, value in modern_grads[:10]]
    
    x = np.arange(len(alexnet_grad_names))
    width = 0.35
    
    ax2.bar(x - width/2, alexnet_grad_values, width, label='AlexNet', alpha=0.7)
    ax2.bar(x + width/2, modern_grad_values[:len(alexnet_grad_names)], width, label='Modern-AlexNet', alpha=0.7)
    ax2.set_title('梯度范数对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('层名称', fontsize=12)
    ax2.set_ylabel('梯度范数', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 死神经元比例对比
    alexnet_dead_ratios = [stats['dead_ratio'] for _, stats in alexnet_acts]
    modern_dead_ratios = [stats['dead_ratio'] for _, stats in modern_acts]
    
    x = np.arange(len(alexnet_dead_ratios))
    ax3.bar(x - width/2, alexnet_dead_ratios, width, label='AlexNet', alpha=0.7)
    ax3.bar(x + width/2, modern_dead_ratios[:len(alexnet_dead_ratios)], width, label='Modern-AlexNet', alpha=0.7)
    ax3.set_title('ReLU死神经元比例对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('ReLU层', fontsize=12)
    ax3.set_ylabel('死神经元比例', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 激活值分布对比
    alexnet_means = [stats['mean'] for _, stats in alexnet_acts]
    modern_means = [stats['mean'] for _, stats in modern_acts]
    
    x = np.arange(len(alexnet_means))
    ax4.bar(x - width/2, alexnet_means, width, label='AlexNet', alpha=0.7)
    ax4.bar(x + width/2, modern_means[:len(alexnet_means)], width, label='Modern-AlexNet', alpha=0.7)
    ax4.set_title('ReLU激活均值对比', fontsize=14, fontweight='bold')
    ax4.set_xlabel('ReLU层', fontsize=12)
    ax4.set_ylabel('激活均值', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('alexnet_analysis.png', dpi=300, bbox_inches='tight')
    print("分析图表已保存为 alexnet_analysis.png")
    
    # 7. 总结问题
    print("\n" + "="*60)
    print("问题总结")
    print("="*60)
    print("AlexNet表现差的主要原因:")
    print("1. LocalResponseNorm层过时且可能导致梯度消失")
    print("2. 权重初始化方法不适合现代优化器")
    print("3. 缺少BatchNorm导致训练不稳定")
    print("4. 模型过于复杂，在少量训练数据上容易过拟合")
    print("5. 学习率可能需要调整")
    print("6. 训练轮数太少，大型模型需要更多时间收敛")
    print("\n建议改进:")
    print("1. 使用Modern-AlexNet（已包含BatchNorm）")
    print("2. 增加训练轮数到10-20个epoch")
    print("3. 调整学习率策略（如学习率衰减）")
    print("4. 使用更好的权重初始化方法")
    print("5. 增加数据增强")
    print("="*60)

if __name__ == '__main__':
    analyze_alexnet_issues()
