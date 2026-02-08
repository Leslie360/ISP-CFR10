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

import argparse
import matplotlib
matplotlib.use('Agg') # 非交互模式
import matplotlib.pyplot as plt

# ==========================================
# 绘图函数 / Plotting
# ==========================================
def plot_training_curves(log_data, save_path):
    """
    绘制训练曲线 (Loss 和 Accuracy)
    log_data: dict, 包含 'train_losses', 'val_losses', 'train_accs', 'val_accs'
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    epochs = range(1, len(log_data['train_losses']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, log_data['train_losses'], 'b-', label='Train Loss')
    plt.plot(epochs, log_data['val_losses'], 'r-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, log_data['train_accs'], 'b-', label='Train Acc')
    plt.plot(epochs, log_data['val_accs'], 'r-', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

def get_scheduler(optimizer, warmup_epochs, max_epochs):
    """
    创建带 Warmup 的余弦退火学习率调度器。
    
    Args:
        optimizer: 优化器
        warmup_epochs: 热身轮数
        max_epochs: 总训练轮数
    """
    # 定义 lambda 函数
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性热身: 从 0 增加到 1
            return float(epoch + 1) / float(max_epochs) if max_epochs == 0 else float(epoch + 1) / float(warmup_epochs)
        else:
            # 余弦退火: 从 1 减少到 0
            # 进度 progress 从 0 到 1
            progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def apply_ltp_ltd_nonlinearity(net):
    ltp_poly = np.poly1d(config.LTP_POLY)
    ltd_poly = np.poly1d(config.LTD_POLY)
    
    # 定义物理影响因子 (0.0 = 纯数字，1.0 = 纯物理)
    # 0.3 是一个经验值，既体现物理特性，又保证绝对稳定
    ALPHA = 0.3 

    for module in net.modules():
        if isinstance(module, model.OrganicSynapseConv):
            if module.weight.grad is not None:
                # 1. 【核心修复】计算斜率前，必须把权重 Clamp 到 [-1, 1]
                # 防止权重飞出范围后，多项式算出一个巨大的错误值
                w_safe = torch.clamp(module.weight.data, -1.0, 1.0)
                
                # 归一化到 [0, 1] 用于计算多项式
                w_norm = (w_safe + 1.0) / 2.0
                
                # 计算物理斜率
                slope_ltp = np.abs(ltp_poly(w_norm.cpu().numpy()))
                slope_ltd = np.abs(ltd_poly(w_norm.cpu().numpy()))
                
                # 归一化斜率 (避免 1e-10 这种数值直接把梯度杀没了)
                # 使用预先计算好的 max 值或动态 max
                # 这里用预先计算好的 max 是因为实验数据是固定的，不会改变
                # 动态 max 会在每次训练时重新计算，可能会引入不稳定因素
                # 所以这里用预先计算好的 max 值
                slope_ltp /= (slope_ltp.max() + 1e-8)
                slope_ltd /= (slope_ltd.max() + 1e-8)
                
                slope_ltp = torch.from_numpy(slope_ltp).to(module.weight.device).float()
                slope_ltd = torch.from_numpy(slope_ltd).to(module.weight.device).float()
                
                # 2. 准备梯度掩码
                grad = module.weight.grad
                mask_ltp = (grad < 0).float()
                mask_ltd = (grad > 0).float()
                
                # 计算物理修正因子
                poly_factor = mask_ltp * slope_ltp + mask_ltd * slope_ltd
                
                # 3. 【残差更新】 
                # New_Grad = Original_Grad * (1 - alpha) + (Original_Grad * Poly_Factor) * alpha
                # 这样即使 Poly_Factor 算错了，至少还有 70% 的原始梯度救命
                scaling = (1 - ALPHA) + (poly_factor * ALPHA)
                
                grad.mul_(scaling)

def train_one_epoch(net, dataloader, criterion, optimizer, device, epoch, scaler):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. 清零梯度
        optimizer.zero_grad()
        
        # ================= [修改] 使用 AMP 上下文 =================
        # 原代码：
        # outputs = net(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # apply_ltp_ltd_nonlinearity(net)
        # optimizer.step()
        
        # 新代码：
        # 使用 bfloat16 (RTX 30/40/50 系列专用，数值更稳)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        
        # 使用 scaler 进行反向传播
        scaler.scale(loss).backward()
        
        #如果你还想保留那个 LTP/LTD 的非线性梯度操作 (apply_ltp_ltd_nonlinearity)
        # 必须先 unscale 梯度，否则梯度是乱的
        scaler.unscale_(optimizer) 
        
        # 在这里调用你的物理非线性函数 (如果有的话)
        apply_ltp_ltd_nonlinearity(net) 
        
        # 梯度裁剪 (可选，防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # 【新增/检查】每次更新完立刻把权重按回 [-1, 1]
        with torch.no_grad():
            for module in net.modules():
                if isinstance(module, model.OrganicSynapseConv):
                     module.weight.data.clamp_(-1.0, 1.0)

        # ===========================================================
        
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Organic Transistor Synapse Training')
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    
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

    # ================= [新增] 2. 启用 torch.compile 加速 =================
    # 这会把你的模型编译成优化的内核，大幅减少 Python 开销
    print("[Infra] Compiling model with torch.compile...")
    try:
        # Windows/WSL 下如果报错，可以把 mode 改为 'default' 或注释掉这行
        net = torch.compile(net, mode='reduce-overhead')
    except Exception as e:
        print(f"[Warning] torch.compile failed: {e}. Continuing without compilation.")
    # ====================================================================

    # 4. 优化器与调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # ================= [新增] 初始化 AMP Scaler =================
    scaler = torch.amp.GradScaler('cuda') 
    # ===========================================================

    # 使用带 Warmup 的调度器
    # 注意：如果从断点恢复，调度器状态也会被加载，这里初始化只是为了创建对象
    scheduler = get_scheduler(optimizer, warmup_epochs=config.WARMUP_EPOCHS, max_epochs=config.EPOCHS)
    
    # 5. 断点续训逻辑 / Resume Training
    start_epoch = 0
    best_acc = 0.0
    
    # 优先使用命令行参数，其次使用 config 中的默认值
    resume_path = args.resume if args.resume else config.RESUME_PATH
    
    if resume_path:
        if os.path.isfile(resume_path):
            print(f"==> Loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 1. 加载模型权重 (必须)
            net.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. 加载优化器状态 (推荐，保留动量)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 3. 更新起始轮数
            start_epoch = checkpoint['epoch']
            
            # 4. 加载最佳准确率 (如果有)
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
            
            # 5. 强制对齐 Scheduler 的步数
            for _ in range(start_epoch):
                scheduler.step()
                
            print(f"==> Loaded checkpoint (epoch {checkpoint['epoch']}, best_acc {best_acc:.2f}%)")
        else:
            print(f"==> No checkpoint found at '{resume_path}'")

    # 6. 训练循环
    logger.info(f"Start Training from epoch {start_epoch+1} to {config.EPOCHS}...")
    
    # 用于可视化的数据记录
    log_data = {
        'train_losses': [], 'val_losses': [],
        'train_accs': [], 'val_accs': []
    }
    
    try:
        for epoch in range(start_epoch, config.EPOCHS):
            start_time = time.time()
            
            train_loss, train_acc = train_one_epoch(net, trainloader, criterion, optimizer, device, epoch, scaler)
            val_loss, val_acc = validate(net, valloader, criterion, device)
            
            scheduler.step()
            
            duration = time.time() - start_time
            
            # 记录数据
            log_data['train_losses'].append(train_loss)
            log_data['val_losses'].append(val_loss)
            log_data['train_accs'].append(train_acc)
            log_data['val_accs'].append(val_acc)
            
            log_msg = (f"Epoch [{epoch+1}/{config.EPOCHS}] "
                       f"Time: {duration:.1f}s | "
                       f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            logger.info(log_msg)
            
            # 保存 Checkpoint (包含恢复所需的所有信息)
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }
            
            # 保存最新的
            latest_path = os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth')
            torch.save(checkpoint_state, latest_path)
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                # 同时也更新最佳模型的 checkpoint 字典，以便可以从最佳点恢复
                checkpoint_state['best_acc'] = best_acc 
                save_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                torch.save(checkpoint_state, save_path)
                logger.info(f"Saved Best Model (Acc: {best_acc:.2f}%) to {save_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        
    finally:
        # 7. 训练结束或中断时绘制曲线
        logger.info("Plotting training curves...")
        viz_path = os.path.join(config.VISUALIZATION_DIR, 'training_curves.png')
        plot_training_curves(log_data, viz_path)
        logger.info(f"Training curves saved to {viz_path}")
        logger.info("Training Finished.")

if __name__ == '__main__':
    main()
