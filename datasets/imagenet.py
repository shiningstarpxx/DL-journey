"""
ImageNet数据集处理模块
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import logging

logger = logging.getLogger(__name__)

class ImageNetDataset:
    """ImageNet数据集处理类"""
    
    def __init__(self, data_dir='./data/imagenet', batch_size=32, num_workers=4, 
                 image_size=224, augment=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.augment = augment
        
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
        self._load_datasets()
    
    def _get_train_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_test_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_datasets(self):
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            logger.warning(f"ImageNet数据集目录不存在: {self.data_dir}")
            logger.info("请下载ImageNet数据集并放置在正确位置")
            raise FileNotFoundError(f"ImageNet数据集目录不存在: {self.data_dir}")
        
        try:
            self.train_dataset = torchvision.datasets.ImageFolder(
                root=train_dir, transform=self.train_transform
            )
            self.test_dataset = torchvision.datasets.ImageFolder(
                root=val_dir, transform=self.test_transform
            )
            logger.info(f"ImageNet数据集加载成功")
            logger.info(f"训练集大小: {len(self.train_dataset)}")
            logger.info(f"验证集大小: {len(self.test_dataset)}")
        except Exception as e:
            logger.error(f"加载ImageNet数据集失败: {e}")
            raise
    
    def get_dataloaders(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        return train_loader, test_loader
    
    def get_class_names(self):
        return self.train_dataset.classes
