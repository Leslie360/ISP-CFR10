---

# Bio-inspired Organic Neuromorphic Vision System (ISP-CFR10)

这是一个基于 PyTorch 的**仿生有机神经形态视觉系统**仿真框架。该项目旨在弥合传统的数字深度学习与新兴的**有机光电计算 (Organic Optoelectronic Computing)** 之间的鸿沟。

本项目构建了一个从“物理光子输入”到“物理并行计算”的全链路仿真环境，并提出了一套**硬件感知的训练算法 (Hardware-Aware Training)**，成功在具有高噪声、非线性的有机电化学晶体管 (OECT) 阵列上实现了 CIFAR-10 数据集的高精度分类 (**~87.3%**)，同时大幅降低了计算能耗。

---

## 🚀 核心创新 (Core Innovations)

### 1. 👁️ 物理感知输入流 (Physics-Aware Input Pipeline)

传统的数字图像（sRGB）无法反映有机传感器面临的真实物理环境。我们构建了逆向 ISP 管道来恢复物理真相：

* **光强反演 (Inverse Gamma)**: 将非线性的 sRGB 信号还原为线性的物理光子通量。
* **光谱响应映射 (Spectral Responsivity)**: 模拟有机半导体材料（如 P3HT:PCBM）对不同波长（R/G/B）的非均匀光电响应特性。
* **泊松散粒噪声 (Poisson Shot Noise)**: 引入符合物理统计规律的光子到达噪声 ()，模拟低光照条件下的真实传感器输出。

### 2. 🧠 有机突触器件仿真 (Organic Synapse Simulation)

我们在卷积层中精确建模了有机电化学晶体管 (OECT) 的物理行为，而非简单的数学乘法：

* **电导映射**: 将神经网络权重映射到器件的物理电导范围 ()。
* **非线性更新 (LTP/LTD)**: 基于真实的实验数据，模拟了器件在写入过程中的长时程增强 (LTP) 和抑制 (LTD) 的非线性动力学及不对称性。
* **器件变异性 (Device Variability)**: 在前向传播中注入高斯噪声，模拟器件的 Cycle-to-Cycle (C2C) 读写噪声。

### 3. 🛡️ 硬件感知训练算法 (Hardware-Aware Training)

针对模拟器件“不精确”的特性，我们设计了鲁棒的训练策略，使神经网络具有“自愈”能力：

* **混合动力梯度更新 (Residual Physics Update)**: 提出 -混合策略，结合数学梯度与物理非线性梯度，防止物理失真导致的训练发散。
* **激进梯度钳制 (Aggressive Gradient Clipping)**: 引入 `max_norm=1.0` 的强力裁剪，防止物理噪声引发的梯度爆炸。
* **硬件约束 (Hard Clamping)**: 在每次更新后强制将权重限制在物理电导的可行域内。

### 4. ⚡ 高性能计算优化 (HPC Optimization)

为了解决物理仿真带来的额外计算开销，我们对训练管线进行了深度优化：

* **混合精度训练 (AMP - BF16)**: 利用 NVIDIA RTX 30/40/50 系列的 **BFloat16** Tensor Cores，在保证物理数值动态范围的同时加速计算。
* **图模式编译 (Torch.compile)**: 使用 PyTorch 2.0 Inductor 后端，将自定义的物理算子融合 (Kernel Fusion)，大幅减少 Python Overhead。
* **大规模并行**: 支持超大 Batch Size (2048)，充分利用显存带宽，显著提升吞吐量。

---

## 📊 性能表现 (Performance)

| Metric | Digital Baseline (FP32) | **Organic Neuromorphic (Ours)** |
| --- | --- | --- |
| **Accuracy (CIFAR-10)** | ~92% | **87.29%** |
| **Bit Precision** | 32-bit Float | **Analog Conductance (Noisy)** |
| **Update Linearity** | Perfect | **Non-linear (LTP/LTD)** |
| **Noise Level** | None | **High (Shot + Device Noise)** |
| **Training Speed** | 1.0x | **~1.5x (with Optimization)** |

> **结果分析**: 尽管引入了严重的物理噪声和非理想特性，我们的硬件感知算法仍能将准确率保持在 87% 以上，证明了有机神经形态计算在边缘智能应用中的巨大潜力。

---

## 🛠️ 环境依赖 (Requirements)

* **Python**: 3.10+
* **PyTorch**: 2.1+ (推荐 2.4+ 以获得最佳 `torch.compile` 支持)
* **CUDA**: 12.1+
* **GPU**: NVIDIA RTX 3060 或更高 (推荐支持 BF16 的显卡)
* **System**: Linux (Ubuntu 22.04+) / WSL2 (推荐)

安装依赖：

```bash
pip install torch torchvision numpy matplotlib
# 确保系统安装了 gcc 以支持 torch.compile
sudo apt install build-essential python3-dev

```

---

## 🏃‍♂️ 快速开始 (Quick Start)

### 1. 训练模型

使用默认配置（模拟真实物理环境 + 硬件感知训练）开始训练：

```bash
python train.py

```

### 2. 从断点恢复

如果训练中断，可以从最新的 Checkpoint 继续（自动重置学习率调度）：

```bash
python train.py --resume ./checkpoints/best_model.pth

```

### 3. 配置物理参数

所有物理参数均在 `config.py` 中定义，可根据不同的有机材料特性进行修改：

```python
# config.py
BATCH_SIZE = 2048        # 针对 RTX 显卡优化
MAX_PHOTONS = 10000      # 调整输入光强 (噪声水平)
LTP_POLY = [...]         # 自定义器件的 LTP 拟合多项式
DEVICE_NOISE_STD = ...   # 调整器件读写噪声

```

---

## 📂 项目结构 (Structure)

* `train.py`: 主训练脚本，包含硬件感知训练逻辑、AMP 混合精度和 Scheduler 管理。
* `model.py`: 定义 ResNet-18 模型及核心的 **`OrganicSynapseConv`** 算子（物理卷积层）。
* `dataset.py`: 自定义数据加载器，实现光强反演、光谱响应和泊松噪声注入。
* `config.py`: 集中管理物理参数、训练超参数和硬件配置。
* `ltp_ltd.txt`: 真实的有机电化学晶体管 (OECT) 实验数据，用于拟合非线性更新曲线。

---

*Created by [Qiao Sir], 2026.02