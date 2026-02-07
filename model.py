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
        # 1. 准备物理参数 (保持不变)
        g_min = config.PHYSICAL_MIN_CURRENT
        g_max = config.PHYSICAL_MAX_CURRENT
        # 动态范围
        dynamic_range = g_max - g_min
        
        # 2. 将标准化权重 [-1, 1] 映射到 物理电流域 [I_min, I_max]
        # w_norm: [-1, 1] -> [0, 1]
        w_zero_one = (torch.clamp(self.weight, -1, 1) + 1.0) / 2.0
        w_physical = w_zero_one * dynamic_range + g_min
        
        # 3. [关键步骤] 注入器件物理噪声 (在物理域进行)
        if self.training:
            # 生成高斯噪声 (单位: Ampere)
            noise = torch.randn_like(w_physical) * config.DEVICE_NOISE_STD
            w_noisy_physical = w_physical + noise
            
            # 物理截断 (电流不可能超过物理极限)
            w_noisy_physical = torch.clamp(w_noisy_physical, g_min, g_max)
        else:
            w_noisy_physical = w_physical

        # 4. [这是你缺失的步骤] 模拟读出电路 (TIA)
        # 将纳安级电流 (1e-9) 重新映射回 神经网络适用的数值范围 [-1, 1]
        # 否则卷积输出会趋近于0，导致梯度消失
        w_normalized_for_calc = (w_noisy_physical - g_min) / dynamic_range # 变回 [0, 1]
        w_normalized_for_calc = w_normalized_for_calc * 2.0 - 1.0        # 变回 [-1, 1]

        # 5. 使用“带噪声但数值正常”的权重进行卷积
        return F.conv2d(input, w_normalized_for_calc, self.bias, self.stride,
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
