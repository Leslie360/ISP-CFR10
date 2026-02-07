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
    Simulate Data-Driven LTP/LTD Non-linearity.
    Modifies gradients based on the slope of real experimental data.
    """
    # Load polynomial coefficients from config
    # P(w) returns the expected delta_w per pulse
    ltp_poly = np.poly1d(config.LTP_POLY)
    ltd_poly = np.poly1d(config.LTD_POLY)
    
    for name, param in net.named_parameters():
        if param.grad is not None and param.requires_grad:
            # We assume convolution weights are the synapses.
            # (Optional: Filter by name if only specific layers are organic)
            
            # 1. Normalize weights to [0, 1] domain to query the polynomial
            # Assuming weights are in range [-1, 1] approximately, we map to [0, 1]
            # w_norm in [0, 1]
            w_norm = (torch.clamp(param.data, -1.0, 1.0) + 1.0) / 2.0
            
            # 2. Calculate scaling factors
            # We use CPU for numpy polyval or implement polyval in torch
            # To be efficient, let's implement polyval in torch
            def torch_polyval(p, x):
                # p is array of coeffs [c_n, ..., c_0]
                val = torch.zeros_like(x)
                for c in p:
                    val = val * x + c
                return val

            # Get coefficients as tensors on the same device
            ltp_coeffs = torch.tensor(config.LTP_POLY, device=param.device, dtype=param.dtype)
            ltd_coeffs = torch.tensor(config.LTD_POLY, device=param.device, dtype=param.dtype)
            
            # Calculate predicted delta (slope)
            # The polynomial predicts "Delta" for a single pulse.
            # We treat this Delta as a scaling factor for the gradient.
            # If the slope is small (saturation), the gradient should be suppressed.
            
            # Note: The polynomials were fitted on *normalized* data [0, 1].
            slope_ltp = torch.abs(torch_polyval(ltp_coeffs, w_norm))
            slope_ltd = torch.abs(torch_polyval(ltd_coeffs, w_norm))
            
            # 3. Apply scaling based on gradient direction
            # If grad < 0, we want to increase weight -> LTP
            # If grad > 0, we want to decrease weight -> LTD
            # (Note: optimizer step is w = w - lr * grad)
            
            # Mask for LTP (Gradient is negative, so update is positive)
            ltp_mask = (param.grad < 0).float()
            # Mask for LTD (Gradient is positive, so update is negative)
            ltd_mask = (param.grad > 0).float()
            
            # Scale gradient
            # We scale the gradient magnitude by the predicted physical slope.
            # We normalize the slope by the maximum possible slope to keep learning rate meaningful?
            # Or we just use it directly. The user said: "determine the Delta W based on current W".
            # The raw polynomial output is ~0.03 (change per pulse in normalized domain).
            # This is quite small. Standard LR is 0.001.
            # If we multiply gradient by this slope directly, the effective LR will be very small.
            # Strategy: Normalize the scaling factor so the *maximum* slope is 1.0.
            # This preserves the max learning rate but enforces the *shape* of the curve.
            
            max_slope_ltp = np.max(np.abs(np.polyval(config.LTP_POLY, np.linspace(0, 1, 100))))
            max_slope_ltd = np.max(np.abs(np.polyval(config.LTD_POLY, np.linspace(0, 1, 100))))
            
            scale_factor = ltp_mask * (slope_ltp / max_slope_ltp) + \
                           ltd_mask * (slope_ltd / max_slope_ltd)
            
            # Apply scaling
            param.grad.data = param.grad.data * scale_factor

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
