import numpy as np
import matplotlib
# 强制使用非交互后端，解决WSL弹窗报错问题
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os

# ==========================================
# 1. 参数设置
# ==========================================
GAMMA = 2.2
# 模拟最大光子数。
# 设置为 300，既能看到噪声，又能保留图像轮廓
MAX_PHOTONS = 50  
OUTPUT_DIR = "./cifar10_rgb_physical_output"

# ==========================================
# 2. 核心算法类 (3通道并行版)
# ==========================================
class PhysicalOpticTransformRGB:
    def __init__(self, gamma=2.2, max_photons=1000):
        self.gamma = gamma
        self.max_photons = max_photons

    def inverse_isp_rgb(self, img_tensor):
        """
        输入: Tensor (3, H, W), 范围 [0, 1]
        输出: Numpy Array (H, W, 3), 范围 [0, 1], 模拟物理光强
        """
        # 调整维度为 (H, W, 3)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # 步骤1: 逆Gamma校正 (整体运算，保留3通道)
        # 此时 img_linear 代表理想情况下的线性光强
        img_linear = np.power(img_np, self.gamma)
        
        # 步骤2: 注入泊松噪声 (针对每个通道独立采样)
        # 计算期望光子数
        expected_photons = img_linear * self.max_photons
        
        # 从泊松分布采样 (模拟光子的离散性)
        # np.random.poisson 支持对整个数组进行并行操作
        noisy_photons = np.random.poisson(np.maximum(expected_photons, 0))
        
        # 步骤3: 归一化回光强/电流信号
        physical_signal = noisy_photons / self.max_photons
        
        # 截断超过1的值(虽然物理上可能溢出，但归一化便于后续计算)
        physical_signal = np.clip(physical_signal, 0, 1.0)
        
        return physical_signal

# ==========================================
# 3. 执行流程
# ==========================================
def run_simulation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    print("正在加载CIFAR-10...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    # 实例化转换器
    converter = PhysicalOpticTransformRGB(gamma=GAMMA, max_photons=MAX_PHOTONS)
    
    print("开始处理前100张图片 (保留RGB通道)...")
    
    converted_data_list = []
    original_preview = []
    physical_preview = []
    
    for i in range(100):
        img, label = trainset[i]
        
        # 执行转换
        rgb_sensor_data = converter.inverse_isp_rgb(img)
        
        # 存入列表
        converted_data_list.append(rgb_sensor_data)
        
        # 收集前5张用于对比
        if i < 5: 
            original_preview.append(img.permute(1, 2, 0).numpy())
            physical_preview.append(rgb_sensor_data)

    # 保存数据: 形状应该是 (100, 32, 32, 3)
    all_data_array = np.array(converted_data_list)
    npy_path = os.path.join(OUTPUT_DIR, "physical_cifar10_rgb_batch100.npy")
    np.save(npy_path, all_data_array)
    
    print(f"数据已保存: {npy_path}")
    print(f"数据形状: {all_data_array.shape} (N, H, W, C)")

    # ==========================================
    # 4. 可视化对比
    # ==========================================
    print("正在生成对比图...")
    plt.figure(figsize=(15, 6))
    
    for i in range(5):
        # 原始 sRGB (这是给人眼在显示器上看的)
        plt.subplot(2, 5, i+1)
        plt.imshow(original_preview[i])
        plt.title(f"Original sRGB #{i}")
        plt.axis('off')
        
        # 物理线性光强 (这是给晶体管“看”的)
        # 注意：由于去掉了Gamma，这行图看起来会比原图“暗”很多
        # 这是物理正确的现象
        plt.subplot(2, 5, i+6)
        plt.imshow(physical_preview[i]) 
        plt.title(f"Physical Linear RGB #{i}")
        plt.axis('off')
        
    img_save_path = os.path.join(OUTPUT_DIR, "rgb_comparison.png")
    plt.tight_layout()
    plt.savefig(img_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化对比图已保存: {img_save_path}")
    print("全部完成。")

if __name__ == "__main__":
    run_simulation()