---

# Organic Transistor Synapse Simulation & Accelerated Training Framework

这是一个基于 PyTorch 的深度学习研究框架，旨在模拟 **有机光电晶体管 (Organic Optoelectronic Transistors)** 作为突触器件在神经网络中的行为。

本项目不仅实现了**材料物理特性的精确仿真**（如光谱响应、散粒噪声、LTP/LTD 非线性），还通过**高性能计算 (HPC)** 技术对训练管线进行了深度优化，实现了从“物理仿真”到“高速训练”的跨越。

---

## 🚀 核心特性 (Key Features)

### 1. 🔬 物理真实性仿真 (Physics-Aware Modeling)

我们将有机半导体器件的物理约束直接嵌入到了数据流和神经网络层中，而非简单的数值计算。

* **光电转换反演 (Inverse ISP & Physical Light)**:
* **逆 Gamma 校正**: 将 sRGB 图像还原为线性物理光强。
* **泊松散粒噪声 (Poisson Shot Noise)**: 模拟光子到达的离散统计特性，引入真实的输入端噪声 ()。
* **光谱响应映射 (Spectral Responsivity)**: 根据有机材料对 R/G/B 不同波长的响应度差异（如绿光响应最强），对输入信号进行加权，模拟真实的载流子生成过程。


* **有机突触卷积层 (Organic Synapse Layer)**:
* **电导映射**: 将神经网络权重映射到真实的纳安级电流范围 ()。
* **器件变异性 (Device Variability)**: 在前向传播中注入高斯噪声，模拟器件的 Cycle-to-cycle 变异。
* **读出电路模拟 (Readout Emulation)**: 实现了模拟-数字信号的重归一化，防止微弱电流导致的梯度消失。



### 2. ⚡ 训练稳定性优化 (Training Stability)

针对物理噪声导致的梯度震荡问题，采用了多种策略确保收敛：

* **硬件感知约束 (Hardware-Aware Constraints)**:
* **Hard Clamping**: 在每次梯度更新后，强制将权重限制在物理定义的电导范围内，防止参数漂移。
* **LTP/LTD 动力学**: (可选) 集成了基于实验数据的长时程增强/抑制非线性更新规则。


* **学习率调度**:
* 采用 `CosineAnnealingLR` 配合 `Warmup` 策略。
* **Warmup**: 在训练初期使用较小学习率预热，适应物理噪声。
* **Cosine Decay**: 后期平滑衰减，帮助模型收敛到平坦的极小值区域。



### 3. 🏎️ 基础设施与加速 (Infrastructure Optimization)

为了解决小模型在高端 GPU (RTX 5070 Ti) 上的利用率低 (GPU Starvation) 问题，我们实施了全栈加速：

* **大规模并行 (Massive Batch Size)**:
* 将 Batch Size 从 `128` 提升至 `2048`，大幅提升计算密度，掩盖 Kernel Launch 开销。
* **Linear Scaling Rule**: 同步调整学习率 () 以匹配大 Batch 训练。


* **混合精度训练 (Mixed Precision / AMP)**:
* 启用 `torch.amp`，使用 **BFloat16 (BF16)** 数据类型。
* 相比 FP16，BF16 拥有与 FP32 相同的动态范围，完美适配物理仿真中跨度极大的数值，同时减少显存占用并利用 Tensor Cores 加速。


* **图模式编译 (JIT Compilation)**:
* 使用 `torch.compile` (Inductor 后端) 对模型进行图层面的算子融合 (Operator Fusion)。
* 消除了 Python 解释器的开销 (Overhead)，显著提升了自定义物理层 (`OrganicSynapseConv`) 的执行效率。


* **数据加载优化**:
* `pin_memory=True`: 锁页内存，加速 CPU 到 GPU 的数据传输。
* `persistent_workers=True`: 避免每个 Epoch 重建数据加载进程。



---

## 🛠️ 环境要求 (Requirements)

* Python 3.10+
* PyTorch 2.0+ (必须支持 `torch.compile` 和 `amp`)
* CUDA 11.8+ / 12.x
* NVIDIA GPU (推荐 RTX 30/40/50 系列以支持 BF16)
* **System Dependencies**: `build-essential`, `python3-dev` (用于 JIT 编译)

---

## 📊 性能对比 (Performance)

| 配置 | Batch Size | Precision | Compiler | GPU Util | Training Speed |
| --- | --- | --- | --- | --- | --- |
| **Baseline** | 128 | FP32 | Eager | ~8% | Slow |
| **Optimized** | 2048 | **BF16** | **Inductor** | **High** | **~10x Faster** |

---

## 🏃‍♂️ 运行指南 (Usage)

### 1. 训练 (Training)

```bash
# 从头开始训练
python train.py

# 从断点继续训练 (Resume)
python train.py --resume ./checkpoints/best_model.pth

```

### 2. 配置 (Configuration)

所有物理参数和训练超参数均在 `config.py` 中定义：

```python
# config.py
BATCH_SIZE = 2048       # 加速关键
MAX_PHOTONS = 10000     # 调节物理噪声水平 (越小噪声越大)
RGB_RESPONSIVITY = ...  # 材料光谱特性

```

---

## 📝 引用与致谢

本项目基于 CIFAR-10 数据集，结合了神经形态计算与高性能深度学习训练技术。代码结构参考了最新的 PyTorch 最佳实践。