import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import config
import dataset
import model

import numpy as np

# ==========================================
# 工具函数 / Utils
# ==========================================
def apply_ltp_ltd_nonlinearity(net):
    """
    仅对 OrganicSynapseConv 层应用基于物理数据的非线性梯度缩放
    """
    # 预先加载系数到 Tensor (避免在循环中重复转换)
    # 假设模型在 GPU，这里创建个 buffer 或者在循环里处理
    # 简单起见，在循环里判断 device
    
    for module in net.modules():
        # [关键修改] 只筛选有机突触层
        if isinstance(module, model.OrganicSynapseConv):
            param = module.weight
            if param.grad is not None:
                # 1. 归一化权重到 [0, 1] 用于查表
                w_norm = (torch.clamp(param.data, -1.0, 1.0) + 1.0) / 2.0
                
                # 2. 准备多项式系数 (确保在同一 Device)
                ltp_coeffs = torch.tensor(config.LTP_POLY, device=param.device, dtype=param.dtype)
                ltd_coeffs = torch.tensor(config.LTD_POLY, device=param.device, dtype=param.dtype)
                
                # 3. 计算多项式值 (物理斜率)
                # 实现简单的 Horner's Method 计算多项式
                def poly_eval(coeffs, x):
                    res = torch.zeros_like(x)
                    for c in coeffs:
                        res = res * x + c
                    return res
                
                slope_ltp = torch.abs(poly_eval(ltp_coeffs, w_norm))
                slope_ltd = torch.abs(poly_eval(ltd_coeffs, w_norm))
                
                # 4. 根据梯度方向选择斜率
                # grad < 0 implies w needs to increase (LTP)
                ltp_mask = (param.grad < 0).float()
                ltd_mask = (param.grad > 0).float()
                
                # 5. 计算缩放因子
                # 为了防止学习率过小，归一化到最大斜率 (Optional, 但推荐)
                # 这里简单处理：直接乘物理斜率可能导致训练极慢(因为数值~1e-10)，
                # 你的 polyfit 是基于 normalized [0,1] data 的，斜率应该是 ~0.03 级别。
                # 建议：归一化斜率，保持最大更新步长由 LR 控制，形状由物理决定。
                max_slope = 0.05 # 估计值，或者动态计算
                scale_factor = (ltp_mask * slope_ltp + ltd_mask * slope_ltd) / max_slope
                
                # 加上一个极小值防止死区
                scale_factor = scale_factor + 0.1 
                
                # 应用缩放
                param.grad.data *= scale_factor

def train_one_epoch(net, dataloader, criterion, optimizer, device, epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. 清零梯度
        optimizer.zero_grad()
        
        # 2. 前向传播 (Forward)
        # 此时 model.OrganicSynapseConv 会自动加入器件噪声 (Device Noise)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # 3. 反向传播 (Backward)
        loss.backward()
        
        # 4. 模拟突触更新非线性 (LTP/LTD Simulation)
        apply_ltp_ltd_nonlinearity(net)
        
        # 5. 更新权重
        optimizer.step()
        
        # [新增] 6. 强制物理约束 (Hard Constraint)
        # 防止权重漂移出物理定义域，模拟器件的硬饱和特性
        with torch.no_grad():
            for module in net.modules():
                if isinstance(module, model.OrganicSynapseConv):
                    module.weight.data.clamp_(-1.0, 1.0)

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(net, dataloader, criterion, device):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 验证时是否需要器件噪声取决于研究目的。
    # 通常为了验证"鲁棒性"，我们希望看到在有噪声的硬件上推理的效果，
    # 所以 model.py 中 OrganicSynapseConv 的 forward 中我设置了始终加噪声 (or self.training check)
    # 检查 model.py logic: if self.training or True -> 始终加噪声。符合验证硬件鲁棒性的目的。
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    # 1. 环境设置
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    # 2. 数据准备
    print("Preparing Data (Physical Optic Simulation + Augmentation)...")
    trainloader, valloader = dataset.get_dataloaders()
    
    # 3. 模型构建 (ResNet18 with OrganicSynapseConv)
    print("Building Model (Organic Transistor Synapse Simulation)...")
    net = model.get_model().to(device)
    
    # 4. 优化器与调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # 5. 训练循环
    best_acc = 0.0
    print(f"Start Training for {config.EPOCHS} epochs...")
    
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(net, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(net, valloader, criterion, device)
        
        scheduler.step()
        
        duration = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Time: {duration:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(net.state_dict(), save_path)
            print(f"Saved Best Model (Acc: {best_acc:.2f}%) to {save_path}")

    print("Training Finished.")

if __name__ == '__main__':
    main()
