import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import config
import dataset
import model

import logging
import sys

# ==========================================
# 日志设置 / Logging Setup
# ==========================================
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.txt")
    
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def log_config_parameters(logger):
    """记录所有配置参数到日志"""
    logger.info("="*40)
    logger.info("训练配置参数 / Training Configuration")
    logger.info("="*40)
    
    # 遍历 config 模块中的变量
    for key in dir(config):
        if not key.startswith("__"):
            value = getattr(config, key)
            # 过滤掉模块引用，只保留数据
            if not isinstance(value, type(sys)):
                logger.info(f"{key}: {value}")
    logger.info("="*40)

# ==========================================
# 工具函数 / Utils
# ==========================================
import numpy as np # Re-import numpy as it was removed

def apply_ltp_ltd_nonlinearity(net):
    """
    模拟数据驱动的 LTP/LTD 非线性。
    根据真实实验数据的斜率修改梯度。
    """
    # 从配置加载多项式系数
    # P(w) 返回每个脉冲的预期 delta_w
    ltp_poly = np.poly1d(config.LTP_POLY)
    ltd_poly = np.poly1d(config.LTD_POLY)
    
    for name, param in net.named_parameters():
        if param.grad is not None and param.requires_grad:
            # 我们假设卷积权重是突触。
            # (可选：如果只有特定层是有机的，则按名称过滤)
            
            # 1. 将权重归一化到 [0, 1] 域以查询多项式
            # 假设权重范围大约为 [-1, 1]，我们将映射到 [0, 1]
            # w_norm 在 [0, 1]
            w_norm = (torch.clamp(param.data, -1.0, 1.0) + 1.0) / 2.0
            
            # 2. 计算缩放因子
            # 我们使用 CPU 进行 numpy polyval 或在 torch 中实现 polyval
            # 为了高效，让我们在 torch 中实现 polyval
            def torch_polyval(p, x):
                # p 是系数数组 [c_n, ..., c_0]
                val = torch.zeros_like(x)
                for c in p:
                    val = val * x + c
                return val

            # 获取相同设备上的系数张量
            ltp_coeffs = torch.tensor(config.LTP_POLY, device=param.device, dtype=param.dtype)
            ltd_coeffs = torch.tensor(config.LTD_POLY, device=param.device, dtype=param.dtype)
            
            # 计算预测的 delta (斜率)
            # 多项式预测单个脉冲的 "Delta"。
            # 我们将此 Delta 视为梯度的缩放因子。
            # 如果斜率很小 (饱和)，梯度应该被抑制。
            
            # 注意：多项式是在 *归一化* 数据 [0, 1] 上拟合的。
            slope_ltp = torch.abs(torch_polyval(ltp_coeffs, w_norm))
            slope_ltd = torch.abs(torch_polyval(ltd_coeffs, w_norm))
            
            # 3. 根据梯度方向应用缩放
            # 如果 grad < 0，我们想要增加权重 -> LTP
            # 如果 grad > 0，我们想要减少权重 -> LTD
            # (注意：优化器步骤是 w = w - lr * grad)
            
            # LTP 掩码 (梯度为负，所以更新为正)
            ltp_mask = (param.grad < 0).float()
            # LTD 掩码 (梯度为正，所以更新为负)
            ltd_mask = (param.grad > 0).float()
            
            # 缩放梯度
            # 我们通过预测的物理斜率缩放梯度幅度。
            # 我们通过最大可能的斜率归一化斜率，以保持学习率有意义？
            # 或者我们直接使用它。用户说："根据当前 W 确定 Delta W"。
            # 原始多项式输出约为 ~0.03 (归一化域中每个脉冲的变化)。
            # 这相当小。标准 LR 为 0.001。
            # 如果我们直接用此斜率乘以梯度，有效 LR 将非常小。
            # 策略：归一化缩放因子，使 *最大* 斜率为 1.0。
            # 这保留了最大学习率，但强制执行曲线的 *形状*。
            
            max_slope_ltp = np.max(np.abs(np.polyval(config.LTP_POLY, np.linspace(0, 1, 100))))
            max_slope_ltd = np.max(np.abs(np.polyval(config.LTD_POLY, np.linspace(0, 1, 100))))
            
            scale_factor = ltp_mask * (slope_ltp / max_slope_ltp) + \
                           ltd_mask * (slope_ltd / max_slope_ltd)
            
            # 应用缩放
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
    
    # 设置日志
    logger = setup_logger(config.LOG_DIR)
    logger.info(f"Using device: {device}")
    
    # 记录配置参数
    log_config_parameters(logger)
    
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    # 2. 数据准备
    logger.info("Preparing Data (Physical Optic Simulation + Augmentation)...")
    trainloader, valloader = dataset.get_dataloaders()
    
    # 3. 模型构建 (ResNet18 with OrganicSynapseConv)
    logger.info("Building Model (Organic Transistor Synapse Simulation)...")
    net = model.get_model().to(device)
    
    # 4. 优化器与调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # 5. 训练循环
    best_acc = 0.0
    logger.info(f"Start Training for {config.EPOCHS} epochs...")
    
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(net, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(net, valloader, criterion, device)
        
        scheduler.step()
        
        duration = time.time() - start_time
        
        log_msg = (f"Epoch [{epoch+1}/{config.EPOCHS}] "
                   f"Time: {duration:.1f}s | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(log_msg)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(net.state_dict(), save_path)
            logger.info(f"Saved Best Model (Acc: {best_acc:.2f}%) to {save_path}")

    logger.info("Training Finished.")

if __name__ == '__main__':
    main()
