"""
设备管理模块
自动检测并使用macOS上的最佳可用设备
"""

import torch
import logging

logger = logging.getLogger(__name__)

def get_best_device():
    """
    获取最佳可用设备
    
    Returns:
        torch.device: 最佳可用设备
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("使用 MPS (Metal Performance Shaders) 设备")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("使用 CUDA 设备")
    else:
        device = torch.device("cpu")
        logger.info("使用 CPU 设备")
    
    return device

def get_device_info():
    """
    获取设备信息
    
    Returns:
        dict: 设备信息字典
    """
    device = get_best_device()
    info = {
        "device": device,
        "device_type": device.type,
        "is_mps": device.type == "mps",
        "is_cuda": device.type == "cuda",
        "is_cpu": device.type == "cpu"
    }
    
    if device.type == "mps":
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
    elif device.type == "cuda":
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
    
    return info

def optimize_for_device(model, device):
    """
    根据设备优化模型
    
    Args:
        model: PyTorch模型
        device: 目标设备
    
    Returns:
        优化后的模型
    """
    model = model.to(device)
    
    if device.type == "mps":
        # MPS特定的优化
        model = model.float()  # MPS目前主要支持float32
    elif device.type == "cpu":
        # CPU优化
        model = model.float()
    
    return model

def get_optimal_batch_size(device, model_size_mb=100):
    """
    根据设备获取最优批处理大小
    
    Args:
        device: 目标设备
        model_size_mb: 模型大小（MB）
    
    Returns:
        int: 推荐的批处理大小
    """
    if device.type == "mps":
        # MPS设备通常内存较大，可以使用较大的批处理
        return 32
    elif device.type == "cuda":
        # CUDA设备根据显存调整
        if torch.cuda.get_device_properties(0).total_memory > 8e9:  # 8GB
            return 64
        else:
            return 32
    else:
        # CPU设备使用较小的批处理
        return 16

def print_device_info():
    """
    打印设备信息
    """
    info = get_device_info()
    print(f"当前设备: {info['device']}")
    print(f"设备类型: {info['device_type']}")
    
    if info['is_mps']:
        print(f"MPS可用: {info['mps_available']}")
        print(f"MPS已构建: {info['mps_built']}")
    elif info['is_cuda']:
        print(f"CUDA设备数量: {info['cuda_device_count']}")
        print(f"CUDA设备名称: {info['cuda_device_name']}")
    
    print(f"推荐批处理大小: {get_optimal_batch_size(info['device'])}")

if __name__ == "__main__":
    print_device_info()
