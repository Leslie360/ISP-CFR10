import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config

class PhysicalOpticTransform:
    """
    模拟物理光学采集过程：
    逆 Gamma 校正 -> 泊松噪声 (光子散粒噪声) -> 归一化
    """
    def __init__(self, gamma=2.2, max_photons=config.MAX_PHOTONS):
        self.gamma = gamma
        self.max_photons = max_photons

    def __call__(self, img_tensor):
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # 1. 逆 Gamma (模拟线性光强)
        img_linear = np.power(img_np, self.gamma)
        
        # 2. [关键修改] 先应用光谱响应，再计算光子/电子数
        # 物理意义：光谱响应度决定了能产生多少有效光电子
        if img_linear.shape[2] == 3:
            responsivity = np.array(config.RGB_RESPONSIVITY)
            # 广播乘法: (H,W,3) * (3,)
            img_effective = img_linear * responsivity
        else:
            img_effective = img_linear

        # 3. 泊松噪声注入 (基于有效光电子数)
        expected_electrons = img_effective * self.max_photons
        noisy_electrons = np.random.poisson(np.maximum(expected_electrons, 0))
        
        # 4. 归一化 (注意：分母依然是 max_photons，保留了响应度带来的变暗效果)
        physical_signal = noisy_electrons / self.max_photons
        physical_signal = np.clip(physical_signal, 0, 1.0)
        
        return torch.from_numpy(physical_signal).permute(2, 0, 1).float()

def get_transforms(train=True):
    """
    返回数据变换流水线。
    """
    # 基础变换
    transform_list = [
        transforms.ToTensor(), # 将 PIL 图片转为 Tensor [0, 1]
        PhysicalOpticTransform(max_photons=config.MAX_PHOTONS), # 物理仿真
    ]
    
    if train:
        # 强力数据增强
        transform_list.extend([
            # 几何变换
            transforms.RandomRotation(15), # 旋转 +/- 15 度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 平移
            
            # 真实场景遮挡模拟
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    
    # 最终归一化 (标准的 CIFAR-10 均值/标准差)
    transform_list.append(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )
    
    return transforms.Compose(transform_list)

def get_dataloaders():
    """
    下载/加载 CIFAR-10 并返回训练/验证数据加载器。
    """
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # 下载并加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root=config.DATA_ROOT, train=True, download=True, transform=train_transform)
    
    trainloader = DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=True)

    # 下载并加载测试集 (此处用作验证集)
    valset = torchvision.datasets.CIFAR10(
        root=config.DATA_ROOT, train=False, download=True, transform=val_transform)
    
    valloader = DataLoader(
        valset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True)

    return trainloader, valloader
