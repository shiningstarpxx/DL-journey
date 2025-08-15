"""
评估指标模块
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def calculate_accuracy(output, target):
    """
    计算准确率
    
    Args:
        output: 模型输出 [batch_size, num_classes]
        target: 真实标签 [batch_size]
        
    Returns:
        float: 准确率
    """
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total

def calculate_loss(output, target, criterion):
    """
    计算损失
    
    Args:
        output: 模型输出
        target: 真实标签
        criterion: 损失函数
        
    Returns:
        float: 损失值
    """
    return criterion(output, target).item()

def calculate_precision_recall_f1(output, target, average='macro'):
    """
    计算精确率、召回率和F1分数
    
    Args:
        output: 模型输出
        target: 真实标签
        average: 平均方式 ('macro', 'micro', 'weighted')
        
    Returns:
        tuple: (precision, recall, f1)
    """
    _, predicted = torch.max(output.data, 1)
    
    # 转换为numpy数组
    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predicted, average=average
    )
    
    return precision, recall, f1

def calculate_confusion_matrix(output, target):
    """
    计算混淆矩阵
    
    Args:
        output: 模型输出
        target: 真实标签
        
    Returns:
        numpy.ndarray: 混淆矩阵
    """
    _, predicted = torch.max(output.data, 1)
    
    # 转换为numpy数组
    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()
    
    return confusion_matrix(target, predicted)

def calculate_top_k_accuracy(output, target, k=5):
    """
    计算Top-K准确率
    
    Args:
        output: 模型输出 [batch_size, num_classes]
        target: 真实标签 [batch_size]
        k: Top-K
        
    Returns:
        float: Top-K准确率
    """
    _, top_k_indices = torch.topk(output, k, dim=1)
    target_expanded = target.unsqueeze(1).expand_as(top_k_indices)
    correct = (top_k_indices == target_expanded).any(dim=1).sum().item()
    total = target.size(0)
    return correct / total

def calculate_class_accuracy(output, target, num_classes):
    """
    计算每个类别的准确率
    
    Args:
        output: 模型输出
        target: 真实标签
        num_classes: 类别数量
        
    Returns:
        dict: 每个类别的准确率
    """
    _, predicted = torch.max(output.data, 1)
    
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for i in range(num_classes):
        mask = (target == i)
        class_correct[i] = (predicted[mask] == target[mask]).sum()
        class_total[i] = mask.sum()
    
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = class_correct[i].item() / class_total[i].item()
        else:
            class_accuracy[i] = 0.0
    
    return class_accuracy

def calculate_metrics_summary(output, target, num_classes, class_names=None):
    """
    计算完整的评估指标摘要
    
    Args:
        output: 模型输出
        target: 真实标签
        num_classes: 类别数量
        class_names: 类别名称列表
        
    Returns:
        dict: 评估指标摘要
    """
    # 基础指标
    accuracy = calculate_accuracy(output, target)
    top5_accuracy = calculate_top_k_accuracy(output, target, k=5)
    
    # 精确率、召回率、F1
    precision, recall, f1 = calculate_precision_recall_f1(output, target)
    
    # 混淆矩阵
    conf_matrix = calculate_confusion_matrix(output, target)
    
    # 每个类别的准确率
    class_accuracy = calculate_class_accuracy(output, target, num_classes)
    
    # 构建摘要
    summary = {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'class_accuracy': class_accuracy
    }
    
    # 如果有类别名称，添加到摘要中
    if class_names:
        summary['class_names'] = class_names
    
    return summary

def print_metrics_summary(summary):
    """
    打印评估指标摘要
    
    Args:
        summary: 评估指标摘要
    """
    print("=" * 50)
    print("评估指标摘要")
    print("=" * 50)
    print(f"准确率: {summary['accuracy']:.4f}")
    print(f"Top-5准确率: {summary['top5_accuracy']:.4f}")
    print(f"精确率: {summary['precision']:.4f}")
    print(f"召回率: {summary['recall']:.4f}")
    print(f"F1分数: {summary['f1']:.4f}")
    
    print("\n混淆矩阵:")
    print(summary['confusion_matrix'])
    
    if 'class_names' in summary:
        print("\n每个类别的准确率:")
        for i, (class_idx, acc) in enumerate(summary['class_accuracy'].items()):
            class_name = summary['class_names'][class_idx] if class_idx < len(summary['class_names']) else f"Class {class_idx}"
            print(f"  {class_name}: {acc:.4f}")
    
    print("=" * 50)
