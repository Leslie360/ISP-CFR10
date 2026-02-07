import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class OrganicSynapseConv(nn.Conv2d):
    """
    模拟有机晶体管突触的自定义卷积层。
    特性：
    1. 权重饱和 (电导范围限制)
    2. 器件变异性 (权重上的高斯噪声)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(OrganicSynapseConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.noise_std = config.DEVICE_NOISE_STD
        # self.g_min, self.g_max = config.CONDUCTANCE_RANGE # 已移除，现在使用物理映射

    def forward(self, input):
        # 1. 权重映射到归一化范围 [0, 1]
        # 我们假设网络学习到的权重在一个较宽的范围内，但我们将它们钳位
        # 以代表有限的电导状态。
        # 由于我们使用标准训练，权重可能为负。
        # 概念上我们将 [-1, 1] (或其他范围) 映射到物理 [Min, Max]。
        # 这里为了简化与提供的物理数据映射，我们钳位到 [0, 1]。
        # (假设网络适应正权重或偏移权重，或者我们将此视为幅值)。
        # 对于标准 ResNet，权重以 0 为中心。
        # 让我们将 [-1, 1] -> [0, 1] 用于注入噪声，然后再映射回来。
        # w_norm = (self.weight.tanh() + 1) / 2 # 平滑映射到 [0, 1]
        # 或者如果我们想强制硬约束，只需钳位并平移。
        # 用户要求："将 ... 映射到物理电流范围 ... 添加 DEVICE_NOISE_STD ... 然后映射回来"
        
        # 让我们使用线性映射策略进行噪声注入：
        # 我们将当前权重值视为"归一化电导"。
        # 我们将在克隆的张量上操作以避免对梯度的原位操作。
        
        # 步骤 A: 将权重钳位到模拟的归一化范围 [-1, 1]
        w_clamped = torch.clamp(self.weight, -1.0, 1.0)
        
        # 步骤 B: 映射到物理电流域 (安培)
        # 我们将 [-1, 1] 映射到 [Min_Current, Max_Current]
        # 为了处理负权重 (抑制性突触)，我们映射幅值？
        # 或者器件是否支持正/负电流？
        # 通常有机晶体管是单极性的 (积累模式)。
        # 为了支持标准神经网络，我们通常使用两个器件 (G+ - G-) 或一个参考。
        # 简化：我们假设噪声添加到权重的 *幅值* 上，与物理电流比例成正比。
        
        # 映射归一化 [-1, 1] -> 物理 [Min, Max]
        # 我们映射范围：
        # 物理跨度 = Max - Min
        # 权重跨度 = 2 (从 -1 到 1)
        scale = (config.PHYSICAL_MAX_CURRENT - config.PHYSICAL_MIN_CURRENT) / 2.0
        bias = (config.PHYSICAL_MAX_CURRENT + config.PHYSICAL_MIN_CURRENT) / 2.0
        
        w_physical = w_clamped * scale + bias # 现在单位是安培 (近似)
        
        # 步骤 C: 添加物理噪声 (安培)
        if self.training or True:
            noise = torch.randn_like(w_physical) * config.DEVICE_NOISE_STD
            w_physical_noisy = w_physical + noise
        else:
            w_physical_noisy = w_physical
            
        # 步骤 D: 映射回权重域
        w_noisy = (w_physical_noisy - bias) / scale
        
        # 使用修改后的权重执行卷积
        return F.conv2d(input, w_noisy, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 使用 OrganicSynapseConv 代替 nn.Conv2d
        self.conv1 = OrganicSynapseConv(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = OrganicSynapseConv(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                OrganicSynapseConv(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        # 添加 Dropout 以防止过拟合
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out) # Add dropout
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out) # Add dropout
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = OrganicSynapseConv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet Layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_model():
    return ResNet18(num_classes=10)
