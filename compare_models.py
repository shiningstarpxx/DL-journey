"""
模型对比脚本
对比AlexNet、Modern-AlexNet、LeNet、Modern-LeNet四个模型在CIFAR-10数据集上的性能
"""

import os
import sys
import torch
import logging
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import font_manager

# 设置中文字体，解决中文显示问题
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'PingFang SC', 'Hiragino Sans GB']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 检查字体是否可用
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
            # 使用第一个可用的中文字体
            font_path = chinese_fonts[0]
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"✅ 已设置中文字体: {font_prop.get_name()}")
        else:
            print("⚠️ 未找到中文字体，图表中的中文可能无法正常显示")
            # 尝试使用系统默认字体
            plt.rcParams['font.family'] = 'sans-serif'
            
    except Exception as e:
        print(f"⚠️ 设置中文字体时出错: {e}")
        print("图表中的中文可能无法正常显示")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparison:
    """模型对比类"""
    
    def __init__(self):
        # 导入项目模块
        from models.alexnet import AlexNet, AlexNetModern
        from models.lenet import LeNet, LeNetModern
        from datasets.cifar import CIFAR10Dataset
        from utils.device import get_best_device, optimize_for_device
        from utils.metrics import calculate_accuracy
        
        # 保存导入的函数
        self.calculate_accuracy = calculate_accuracy
        
        # 强制使用CPU以避免MPS兼容性问题
        self.device = torch.device("cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 创建数据集（使用较小的batch_size和更少的数据）
        logger.info("📊 加载CIFAR-10数据集...")
        self.dataset = CIFAR10Dataset(batch_size=64, num_workers=0)
        self.train_loader, self.test_loader = self.dataset.get_dataloaders()
        
        # 创建四个模型
        logger.info("📦 创建模型...")
        self.models = {
            'AlexNet': AlexNet(num_classes=10),
            'Modern-AlexNet': AlexNetModern(num_classes=10),
            'LeNet': LeNet(num_classes=10, input_channels=3),  # 支持彩色图像
            'Modern-LeNet': LeNetModern(num_classes=10, input_channels=3)
        }
        
        # 优化模型到设备
        for name, model in self.models.items():
            self.models[name] = optimize_for_device(model, self.device)
            logger.info(f"{name}参数数量: {model.count_parameters():,}")
            logger.info(f"{name}模型大小: {model.get_model_size_mb():.2f} MB")
        
        # 训练配置（减少训练时间）
        self.epochs = 3  # 减少到3个epoch
        self.learning_rate = 0.001
        
        # 存储训练历史
        self.history = {
            name: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            for name in self.models.keys()
        }
    
    def train_model(self, model, model_name):
        """训练单个模型"""
        logger.info(f"🎯 开始训练 {model_name}...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(self.epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            # 只训练部分数据以节省时间
            train_iter = iter(self.train_loader)
            num_batches = min(100, len(self.train_loader))  # 限制训练批次数量
            
            progress_bar = tqdm(range(num_batches), desc=f'{model_name} Epoch {epoch+1}/{self.epochs}')
            
            for _ in progress_bar:
                try:
                    data, target = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    data, target = next(train_iter)
                
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_correct += self.calculate_accuracy(output, target) * target.size(0)
                total_samples += target.size(0)
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{self.calculate_accuracy(output, target):.2%}'
                })
            
            avg_train_loss = total_loss / num_batches
            avg_train_acc = total_correct / total_samples
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            
            # 验证阶段
            model.eval()
            total_val_loss = 0
            total_val_correct = 0
            total_val_samples = 0
            
            with torch.no_grad():
                # 只验证部分数据
                val_iter = iter(self.test_loader)
                num_val_batches = min(50, len(self.test_loader))
                
                for _ in range(num_val_batches):
                    try:
                        data, target = next(val_iter)
                    except StopIteration:
                        val_iter = iter(self.test_loader)
                        data, target = next(val_iter)
                    
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    total_val_loss += loss.item()
                    total_val_correct += self.calculate_accuracy(output, target) * target.size(0)
                    total_val_samples += target.size(0)
            
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_acc = total_val_correct / total_val_samples
            val_losses.append(avg_val_loss)
            val_accs.append(avg_val_acc)
            
            logger.info(f'{model_name} Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2%}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2%}')
        
        return train_losses, train_accs, val_losses, val_accs
    
    def compare_models(self):
        """对比模型性能"""
        logger.info("🚀 开始模型对比...")
        
        # 训练所有模型
        for model_name, model in self.models.items():
            train_loss, train_acc, val_loss, val_acc = self.train_model(model, model_name)
            
            # 存储结果
            self.history[model_name]['train_loss'] = train_loss
            self.history[model_name]['train_acc'] = train_acc
            self.history[model_name]['val_loss'] = val_loss
            self.history[model_name]['val_acc'] = val_acc
        
        # 最终评估
        self.final_evaluation()
        
        # 绘制对比图
        self.plot_comparison()
        
        # 保存结果
        self.save_results()
    
    def final_evaluation(self):
        """最终评估"""
        logger.info("🔍 最终评估...")
        
        results = {}
        
        for model_name, model in self.models.items():
            model.eval()
            total_correct = 0
            total_samples = 0
            inference_times = []
            
            with torch.no_grad():
                # 只评估部分测试数据
                test_iter = iter(self.test_loader)
                num_test_batches = min(100, len(self.test_loader))
                
                for _ in tqdm(range(num_test_batches), desc=f'Evaluating {model_name}'):
                    try:
                        data, target = next(test_iter)
                    except StopIteration:
                        test_iter = iter(self.test_loader)
                        data, target = next(test_iter)
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 测量推理时间
                    start_time = time.time()
                    output = model(data)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    total_correct += self.calculate_accuracy(output, target) * target.size(0)
                    total_samples += target.size(0)
            
            accuracy = total_correct / total_samples
            avg_inference_time = np.mean(inference_times)
            
            results[model_name] = {
                'final_accuracy': accuracy,
                'avg_inference_time': avg_inference_time,
                'model_params': model.count_parameters(),
                'model_size_mb': model.get_model_size_mb()
            }
            
            logger.info(f"{model_name} - 最终准确率: {accuracy:.2%}, 平均推理时间: {avg_inference_time:.4f}s")
        
        self.final_results = results
    
    def plot_comparison(self):
        """绘制对比图"""
        logger.info("📊 绘制对比图...")
        
        # 设置中文字体
        setup_chinese_font()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, self.epochs + 1)
        colors = ['blue', 'red', 'green', 'orange']
        
        # 训练损失对比
        for i, (name, history) in enumerate(self.history.items()):
            ax1.plot(epochs, history['train_loss'], color=colors[i], label=name, linewidth=2, marker='o')
        ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 训练准确率对比
        for i, (name, history) in enumerate(self.history.items()):
            ax2.plot(epochs, history['train_acc'], color=colors[i], label=name, linewidth=2, marker='o')
        ax2.set_title('训练准确率对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 验证损失对比
        for i, (name, history) in enumerate(self.history.items()):
            ax3.plot(epochs, history['val_loss'], color=colors[i], label=name, linewidth=2, marker='o')
        ax3.set_title('验证损失对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 验证准确率对比
        for i, (name, history) in enumerate(self.history.items()):
            ax4.plot(epochs, history['val_acc'], color=colors[i], label=name, linewidth=2, marker='o')
        ax4.set_title('验证准确率对比', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("对比图已保存为 model_comparison.png")
        
        # 绘制性能对比柱状图
        self.plot_performance_comparison()
    
    def plot_performance_comparison(self):
        """绘制性能对比柱状图"""
        logger.info("📊 绘制性能对比柱状图...")
        
        # 设置中文字体
        setup_chinese_font()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(self.final_results.keys())
        accuracies = [self.final_results[name]['final_accuracy'] for name in model_names]
        inference_times = [self.final_results[name]['avg_inference_time'] for name in model_names]
        params = [self.final_results[name]['model_params'] for name in model_names]
        sizes = [self.final_results[name]['model_size_mb'] for name in model_names]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        
        # 准确率对比
        bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8)
        ax1.set_title('最终准确率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, max(accuracies) * 1.1)
        # 在柱子上添加数值标签
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 推理时间对比
        bars2 = ax2.bar(model_names, inference_times, color=colors, alpha=0.8)
        ax2.set_title('平均推理时间对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        # 在柱子上添加数值标签
        for bar, time_val in zip(bars2, inference_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        # 参数数量对比
        bars3 = ax3.bar(model_names, params, color=colors, alpha=0.8)
        ax3.set_title('模型参数数量对比', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Parameters', fontsize=12)
        # 在柱子上添加数值标签
        for bar, param in zip(bars3, params):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(params) * 0.01,
                    f'{param:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 模型大小对比
        bars4 = ax4.bar(model_names, sizes, color=colors, alpha=0.8)
        ax4.set_title('模型大小对比', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Size (MB)', fontsize=12)
        # 在柱子上添加数值标签
        for bar, size in zip(bars4, sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{size:.2f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 旋转x轴标签以避免重叠
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("性能对比图已保存为 performance_comparison.png")
    
    def save_results(self):
        """保存结果"""
        logger.info("💾 保存结果...")
        
        results = {
            'final_results': self.final_results,
            'training_history': self.history,
            'model_info': {
                name: {
                    'parameters': model.count_parameters(),
                    'size_mb': model.get_model_size_mb()
                }
                for name, model in self.models.items()
            }
        }
        
        with open('comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("结果已保存为 comparison_results.json")
    
    def print_summary(self):
        """打印对比摘要"""
        print("\n" + "="*80)
        print("模型对比摘要")
        print("="*80)
        
        # 创建表格头部
        print(f"{'模型名称':<15} {'准确率':<10} {'推理时间(s)':<12} {'参数数量':<12} {'模型大小(MB)':<12}")
        print("-" * 80)
        
        # 按准确率排序
        sorted_results = sorted(self.final_results.items(), 
                              key=lambda x: x[1]['final_accuracy'], reverse=True)
        
        for model_name, results in sorted_results:
            acc = results['final_accuracy']
            time_val = results['avg_inference_time']
            params = results['model_params']
            size = results['model_size_mb']
            
            print(f"{model_name:<15} {acc:.2%} {time_val:.4f}s {params:,} {size:.2f}")
        
        print("-" * 80)
        
        # 找出最佳模型
        best_model = sorted_results[0]
        print(f"\n🏆 最佳准确率模型: {best_model[0]} ({best_model[1]['final_accuracy']:.2%})")
        
        # 找出最快模型
        fastest_model = min(self.final_results.items(), 
                          key=lambda x: x[1]['avg_inference_time'])
        print(f"⚡ 最快推理模型: {fastest_model[0]} ({fastest_model[1]['avg_inference_time']:.4f}s)")
        
        # 找出最小模型
        smallest_model = min(self.final_results.items(), 
                           key=lambda x: x[1]['model_size_mb'])
        print(f"📦 最小模型: {smallest_model[0]} ({smallest_model[1]['model_size_mb']:.2f}MB)")
        
        print("\n结论:")
        print("1. 现代版本模型通常具有更好的性能和更快的收敛速度")
        print("2. AlexNet系列模型参数更多，但通常能获得更高的准确率")
        print("3. LeNet系列模型更轻量，推理速度更快")
        print("4. 现代版本通过BatchNorm等技术提高了训练稳定性")
        print("="*80)

def main():
    """主函数"""
    try:
        # 确保在虚拟环境中运行
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"MPS可用: {torch.backends.mps.is_available()}")
        
        # 创建对比器并运行对比
        comparator = ModelComparison()
        comparator.compare_models()
        comparator.print_summary()
        
        print("\n🎉 模型对比完成！")
        print("📊 查看 model_comparison.png 获取训练曲线对比")
        print("📊 查看 performance_comparison.png 获取性能对比")
        print("📄 查看 comparison_results.json 获取详细数据")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已激活虚拟环境并安装了所有依赖")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
