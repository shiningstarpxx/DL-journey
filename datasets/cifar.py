"""
CIFAR数据集处理模块
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class CIFAR10Dataset:
    """CIFAR-10数据集处理类"""
    
    def __init__(self, data_dir='./data', batch_size=32, num_workers=4, 
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
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_datasets(self):
        try:
            self.train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=self.train_transform
            )
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.test_transform
            )
            logger.info(f"CIFAR-10数据集加载成功")
        except Exception as e:
            logger.error(f"加载CIFAR-10数据集失败: {e}")
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
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CIFAR100Dataset:
    """CIFAR-100数据集处理类"""
    
    def __init__(self, data_dir='./data', batch_size=32, num_workers=4, 
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
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
    
    def _get_test_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
    
    def _load_datasets(self):
        try:
            self.train_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=self.train_transform
            )
            self.test_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, download=True, transform=self.test_transform
            )
            logger.info(f"CIFAR-100数据集加载成功")
        except Exception as e:
            logger.error(f"加载CIFAR-100数据集失败: {e}")
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
        return torchvision.datasets.CIFAR100.classes
