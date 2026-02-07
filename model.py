import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class OrganicSynapseConv(nn.Conv2d):
    """
    Custom Convolution Layer simulating Organic Transistor Synapses.
    Features:
    1. Weight Saturation (Conductance Range)
    2. Device Variability (Gaussian Noise on Weights)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(OrganicSynapseConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.noise_std = config.DEVICE_NOISE_STD
        # self.g_min, self.g_max = config.CONDUCTANCE_RANGE # Removed, using physical mapping now

    def forward(self, input):
        # 1. Weight Mapping to Normalized Range [0, 1]
        # We assume the network learns weights in a broader range, but we clamp them 
        # to represent the finite conductance states.
        # Since we use standard training, weights can be negative. 
        # We map [-1, 1] (or whatever range) to physical [Min, Max] conceptually.
        # Here we just clamp to [0, 1] for simplicity of mapping to the physical data provided.
        # (Assuming the network adapts to positive-only or shifted weights if necessary, 
        # or we treat this as magnitude).
        # For a standard ResNet, weights are centered at 0.
        # Let's map [-1, 1] -> [0, 1] for noise injection, then map back.
        # w_norm = (self.weight.tanh() + 1) / 2 # Smooth mapping to [0, 1]
        # OR simply clamp and shift if we want to enforce hard constraints.
        # User requirement: "Map ... to physical current range ... to add DEVICE_NOISE_STD ... then map back"
        
        # Let's use a linear mapping strategy for noise injection:
        # We treat the current weight value as "normalized conductance".
        # We'll work on a cloned tensor to avoid in-place ops on gradients.
        
        # Step A: Clamp weights to simulated normalized range [-1, 1]
        w_clamped = torch.clamp(self.weight, -1.0, 1.0)
        
        # Step B: Map to Physical Current Domain (Amperes)
        # We map [-1, 1] to [Min_Current, Max_Current]
        # To handle negative weights (inhibitory synapses), we map magnitude?
        # Or does the device support positive/negative currents?
        # Usually organic transistors are unipolar (accumulation mode).
        # To support standard NN, we typically use two devices (G+ - G-) or a reference.
        # SIMPLIFICATION: We assume the noise is added to the *magnitude* of the weight 
        # proportional to the physical current scale.
        
        # Map Normalized [-1, 1] -> Physical [Min, Max]
        # We map the range:
        # Physical Span = Max - Min
        # Weight Span = 2 (from -1 to 1)
        scale = (config.PHYSICAL_MAX_CURRENT - config.PHYSICAL_MIN_CURRENT) / 2.0
        bias = (config.PHYSICAL_MAX_CURRENT + config.PHYSICAL_MIN_CURRENT) / 2.0
        
        w_physical = w_clamped * scale + bias # Now in Amperes (approx)
        
        # Step C: Add Physical Noise (in Amperes)
        if self.training or True:
            noise = torch.randn_like(w_physical) * config.DEVICE_NOISE_STD
            w_physical_noisy = w_physical + noise
        else:
            w_physical_noisy = w_physical
            
        # Step D: Map back to Weight Domain
        w_noisy = (w_physical_noisy - bias) / scale
        
        # Perform convolution with the modified weights
        return F.conv2d(input, w_noisy, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Use OrganicSynapseConv instead of nn.Conv2d
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
        
        # Dropout to prevent overfitting as requested
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
