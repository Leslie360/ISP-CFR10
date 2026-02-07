import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config

class PhysicalOpticTransform:
    """
    Simulates the physical optical acquisition process:
    Inverse Gamma -> Poisson Noise (Photon Shot Noise) -> Normalization
    """
    def __init__(self, gamma=2.2, max_photons=config.MAX_PHOTONS):
        self.gamma = gamma
        self.max_photons = max_photons

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (Tensor): Input image tensor (C, H, W) in range [0, 1].
        Returns:
            Tensor: Noisy image tensor (C, H, W) in range [0, 1].
        """
        # Convert to numpy (C, H, W) -> (H, W, C) for consistent processing with ISP logic
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # 1. Inverse Gamma Correction (Simulate linear light intensity)
        img_linear = np.power(img_np, self.gamma)
        
        # 2. Poisson Noise Injection
        expected_photons = img_linear * self.max_photons
        # Poisson sampling (simulating discrete photon arrival)
        noisy_photons = np.random.poisson(np.maximum(expected_photons, 0))
        
        # 3. Normalize back to signal range
        physical_signal = noisy_photons / self.max_photons
        physical_signal = np.clip(physical_signal, 0, 1.0)
        
        # 4. Spectral Responsivity Simulation
        # Multiply each channel by its responsivity coefficient
        # physical_signal shape is (H, W, C) where C=3 (RGB)
        if physical_signal.shape[2] == 3:
            responsivity = np.array(config.RGB_RESPONSIVITY)
            physical_signal = physical_signal * responsivity
        
        # Convert back to Tensor (C, H, W)
        return torch.from_numpy(physical_signal).permute(2, 0, 1).float()

def get_transforms(train=True):
    """
    Returns the data transformation pipeline.
    """
    # Base transforms
    transform_list = [
        transforms.ToTensor(), # Convert PIL to Tensor [0, 1]
        PhysicalOpticTransform(max_photons=config.MAX_PHOTONS), # Physical simulation
    ]
    
    if train:
        # Strong Data Augmentation
        transform_list.extend([
            # Geometric Transformations
            transforms.RandomRotation(15), # Rotate +/- 15 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Translation
            
            # Real-world Occlusion Simulation
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    
    # Final Normalization (Standard CIFAR-10 mean/std)
    transform_list.append(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )
    
    return transforms.Compose(transform_list)

def get_dataloaders():
    """
    Downloads/Loads CIFAR-10 and returns train/val dataloaders.
    """
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Download and load training set
    trainset = torchvision.datasets.CIFAR10(
        root=config.DATA_ROOT, train=True, download=True, transform=train_transform)
    
    trainloader = DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=True)

    # Download and load test set (used as validation here)
    valset = torchvision.datasets.CIFAR10(
        root=config.DATA_ROOT, train=False, download=True, transform=val_transform)
    
    valloader = DataLoader(
        valset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True)

    return trainloader, valloader
