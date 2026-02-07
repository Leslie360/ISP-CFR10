import torch
import numpy as np

# ==========================================
# 路径设置 / Paths
# ==========================================
DATA_ROOT = './data'
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'
RESUME_PATH = None
VISUALIZATION_DIR = './results'

# ==========================================
# 训练超参数 / Training Hyperparameters
# ==========================================
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
EPOCHS = 150
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8 # 稍微给多点 CPU 线程
WARMUP_EPOCHS = 10 # 热身轮数

# ==========================================
# 物理参数 (有机晶体管) / Physical Parameters
# ==========================================
# 1. 模拟光子噪声的光强 (Poisson Noise)
MAX_PHOTONS = 10000  

# 2. 光谱响应系数 (RGB Responsivity)
# 输入通道 *= 响应系数
RGB_RESPONSIVITY = [0.706, 1.0, 0.758] # [R, G, B]

# 3. 真实 LTP/LTD 响应数据 (绝对电流值，单位：安培)
REAL_LTP_DATA = np.abs([ 
    -1.59E-10, -3.95E-10, -6.12E-10, -8.27E-10, -1.00E-09, -1.16E-09, 
    -1.30E-09, -1.41E-09, -1.51E-09, -1.58E-09, -1.63E-09, -1.67E-09, 
    -1.71E-09, -1.75E-09, -1.78E-09, -1.81E-09, -1.84E-09, -1.86E-09, 
    -1.88E-09, -1.89E-09, -1.90E-09, -1.91E-09, -1.91E-09, -1.92E-09, 
    -1.92E-09, -1.93E-09, -1.93E-09, -1.94E-09, -1.94E-09, -1.94E-09 
]) 

REAL_LTD_DATA = np.abs([ 
    -1.51E-09, -1.18E-09, -9.99E-10, -8.83E-10, -7.71E-10, -6.68E-10, 
    -5.69E-10, -5.00E-10, -4.47E-10, -4.04E-10, -3.72E-10, -3.41E-10, 
    -3.17E-10, -2.95E-10, -2.79E-10, -2.63E-10, -2.49E-10, -2.37E-10, 
    -2.27E-10, -2.18E-10, -2.11E-10, -2.02E-10, -1.96E-10, -1.89E-10, 
    -1.82E-10, -1.74E-10, -1.72E-10, -1.67E-10, -1.62E-10 
]) 

# 4. 电导范围映射 / Conductance Range Mapping
# 神经网络权重 [0, 1] <-> 物理电流 [Min, Max]
# 注意：由于网络权重通常以 0 为中心 (例如 He 初始化)，
# 我们可能会映射 [-1, 1] 或使用仅正约束。
# 然而，用户提到 "Normalize this data to a [0, 1] range"。
# 标准 Conv2d 权重是有符号的。
# 策略：我们假设物理器件代表幅值或偏移版本。
# 为了简化 `model.py`，我们将 *归一化* 权重 [0, 1] 映射到物理电流。
PHYSICAL_MIN_CURRENT = np.min(REAL_LTP_DATA) # 约 1.59e-10 
PHYSICAL_MAX_CURRENT = np.max(REAL_LTP_DATA) # 约 1.94e-9 

# 器件循环变异性 (动态范围的 2%) / Device Cycle-to-Cycle Variation
DEVICE_NOISE_STD = (PHYSICAL_MAX_CURRENT - PHYSICAL_MIN_CURRENT) * 0.02

# ==========================================
# 更新规则的多项式拟合 / Polynomial Fitting for Update Rules
# ==========================================
# 我们想要建模 dG/dPulse vs G (归一化)。
# 1. 归一化数据到 [0, 1]
_ltp_norm = (REAL_LTP_DATA - PHYSICAL_MIN_CURRENT) / (PHYSICAL_MAX_CURRENT - PHYSICAL_MIN_CURRENT)
_ltd_norm = (REAL_LTD_DATA - PHYSICAL_MIN_CURRENT) / (PHYSICAL_MAX_CURRENT - PHYSICAL_MIN_CURRENT)

# 2. 计算 Delta (每个脉冲的变化)
# 我们假设数据点是连续脉冲。
_delta_ltp = np.diff(_ltp_norm)
_delta_ltd = np.diff(_ltd_norm)

# 3. 拟合多项式: Delta = P(Current_Value)
# 对于 LTP，我们根据当前值 REAL_LTP_DATA[:-1] 预测 delta
# 对于 LTD，我们根据当前值 REAL_LTD_DATA[:-1] 预测 delta
# 3 次多项式应该足以捕捉饱和特性。
LTP_POLY = np.polyfit(_ltp_norm[:-1], _delta_ltp, 3)
LTD_POLY = np.polyfit(_ltd_norm[:-1], _delta_ltd, 3)

# ==========================================
# 硬件/系统 / Hardware/System
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4
