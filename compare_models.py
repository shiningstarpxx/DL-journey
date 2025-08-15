"""
模型对比脚本
同时训练AlexNet和LeNet，对比它们在CIFAR-10数据集上的性能
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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparison:
    """模型对比类"""
    
    def __init__(self):
        # 导入项目模块
        from models.alexnet import create_alexnet
        from models.lenet import create_lenet
        from datasets.cifar import CIFAR10Dataset
        from utils.device import get_best_device, optimize_for_device
        from utils.metrics import calculate_accuracy
        
        # 保存导入的函数
        self.calculate_accuracy = calculate_accuracy
        
        self.device = get_best_device()
        logger.info(f"使用设备: {self.device}")
        
        # 创建数据集
        logger.info("📊 加载CIFAR-10数据集...")
        self.dataset = CIFAR10Dataset(batch_size=32, num_workers=0)
        self.train_loader, self.test_loader = self.dataset.get_dataloaders()
        
        # 创建模型
        logger.info("📦 创建模型...")
        self.alexnet = create_alexnet(num_classes=10, modern=True)
        self.alexnet = optimize_for_device(self.alexnet, self.device)
        
        self.lenet = create_lenet(num_classes=10, modern=True, input_channels=3)
        self.lenet = optimize_for_device(self.lenet, self.device)
        
        # 打印模型信息
        logger.info(f"AlexNet参数数量: {self.alexnet.count_parameters():,}")
        logger.info(f"LeNet参数数量: {self.lenet.count_parameters():,}")
        logger.info(f"AlexNet模型大小: {self.alexnet.get_model_size_mb():.2f} MB")
        logger.info(f"LeNet模型大小: {self.lenet.get_model_size_mb():.2f} MB")
        
        # 训练配置
        self.epochs = 5  # 对比训练5个epoch
        self.learning_rate = 0.001
        
        # 存储训练历史
        self.history = {
            'alexnet': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            'lenet': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
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
            
            progress_bar = tqdm(self.train_loader, desc=f'{model_name} Epoch {epoch+1}/{self.epochs}')
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_correct += self.calculate_accuracy(output, target) * target.size(0)
                total_samples += target.size(0)
                
                if len(progress_bar) % 100 == 0:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{self.calculate_accuracy(output, target):.2%}'
                    })
            
            avg_train_loss = total_loss / len(self.train_loader)
            avg_train_acc = total_correct / total_samples
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            
            # 验证阶段
            model.eval()
            total_val_loss = 0
            total_val_correct = 0
            total_val_samples = 0
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    total_val_loss += loss.item()
                    total_val_correct += self.calculate_accuracy(output, target) * target.size(0)
                    total_val_samples += target.size(0)
            
            avg_val_loss = total_val_loss / len(self.test_loader)
            avg_val_acc = total_val_correct / total_val_samples
            val_losses.append(avg_val_loss)
            val_accs.append(avg_val_acc)
            
            logger.info(f'{model_name} Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2%}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2%}')
        
        return train_losses, train_accs, val_losses, val_accs
    
    def compare_models(self):
        """对比模型性能"""
        logger.info("🚀 开始模型对比...")
        
        # 训练AlexNet
        alexnet_train_loss, alexnet_train_acc, alexnet_val_loss, alexnet_val_acc = self.train_model(self.alexnet, "AlexNet")
        
        # 训练LeNet
        lenet_train_loss, lenet_train_acc, lenet_val_loss, lenet_val_acc = self.train_model(self.lenet, "LeNet")
        
        # 存储结果
        self.history['alexnet']['train_loss'] = alexnet_train_loss
        self.history['alexnet']['train_acc'] = alexnet_train_acc
        self.history['alexnet']['val_loss'] = alexnet_val_loss
        self.history['alexnet']['val_acc'] = alexnet_val_acc
        
        self.history['lenet']['train_loss'] = lenet_train_loss
        self.history['lenet']['train_acc'] = lenet_train_acc
        self.history['lenet']['val_loss'] = lenet_val_loss
        self.history['lenet']['val_acc'] = lenet_val_acc
        
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
        
        for model_name, model in [('AlexNet', self.alexnet), ('LeNet', self.lenet)]:
            model.eval()
            total_correct = 0
            total_samples = 0
            inference_times = []
            
            with torch.no_grad():
                for data, target in tqdm(self.test_loader, desc=f'Evaluating {model_name}'):
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, self.epochs + 1)
        
        # 训练损失对比
        ax1.plot(epochs, self.history['alexnet']['train_loss'], 'b-', label='AlexNet', linewidth=2)
        ax1.plot(epochs, self.history['lenet']['train_loss'], 'r-', label='LeNet', linewidth=2)
        ax1.set_title('训练损失对比')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 训练准确率对比
        ax2.plot(epochs, self.history['alexnet']['train_acc'], 'b-', label='AlexNet', linewidth=2)
        ax2.plot(epochs, self.history['lenet']['train_acc'], 'r-', label='LeNet', linewidth=2)
        ax2.set_title('训练准确率对比')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 验证损失对比
        ax3.plot(epochs, self.history['alexnet']['val_loss'], 'b-', label='AlexNet', linewidth=2)
        ax3.plot(epochs, self.history['lenet']['val_loss'], 'r-', label='LeNet', linewidth=2)
        ax3.set_title('验证损失对比')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        # 验证准确率对比
        ax4.plot(epochs, self.history['alexnet']['val_acc'], 'b-', label='AlexNet', linewidth=2)
        ax4.plot(epochs, self.history['lenet']['val_acc'], 'r-', label='LeNet', linewidth=2)
        ax4.set_title('验证准确率对比')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("对比图已保存为 model_comparison.png")
    
    def save_results(self):
        """保存结果"""
        logger.info("💾 保存结果...")
        
        results = {
            'final_results': self.final_results,
            'training_history': self.history,
            'model_info': {
                'alexnet': {
                    'parameters': self.alexnet.count_parameters(),
                    'size_mb': self.alexnet.get_model_size_mb()
                },
                'lenet': {
                    'parameters': self.lenet.count_parameters(),
                    'size_mb': self.lenet.get_model_size_mb()
                }
            }
        }
        
        with open('comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("结果已保存为 comparison_results.json")
    
    def print_summary(self):
        """打印对比摘要"""
        print("\n" + "="*60)
        print("模型对比摘要")
        print("="*60)
        
        print(f"{'指标':<15} {'AlexNet':<15} {'LeNet':<15} {'差异':<15}")
        print("-" * 60)
        
        alexnet_acc = self.final_results['AlexNet']['final_accuracy']
        lenet_acc = self.final_results['LeNet']['final_accuracy']
        acc_diff = alexnet_acc - lenet_acc
        
        alexnet_time = self.final_results['AlexNet']['avg_inference_time']
        lenet_time = self.final_results['LeNet']['avg_inference_time']
        time_diff = lenet_time - alexnet_time
        
        alexnet_params = self.final_results['AlexNet']['model_params']
        lenet_params = self.final_results['LeNet']['model_params']
        params_ratio = alexnet_params / lenet_params
        
        print(f"{'最终准确率':<15} {alexnet_acc:.2%} {lenet_acc:.2%} {acc_diff:+.2%}")
        print(f"{'推理时间(s)':<15} {alexnet_time:.4f} {lenet_time:.4f} {time_diff:+.4f}")
        print(f"{'参数数量':<15} {alexnet_params:,} {lenet_params:,} {params_ratio:.1f}x")
        
        print("\n结论:")
        if acc_diff > 0:
            print(f"✅ AlexNet在准确率上表现更好 (+{acc_diff:.2%})")
        else:
            print(f"✅ LeNet在准确率上表现更好 ({acc_diff:.2%})")
        
        if time_diff > 0:
            print(f"✅ AlexNet推理速度更快 (-{time_diff:.4f}s)")
        else:
            print(f"✅ LeNet推理速度更快 (+{abs(time_diff):.4f}s)")
        
        print(f"✅ AlexNet参数数量是LeNet的 {params_ratio:.1f} 倍")
        print("="*60)

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
        print("📊 查看 model_comparison.png 获取可视化结果")
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
