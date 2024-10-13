import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class WaveletCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, wavelet='db1'):
        super(WaveletCNNBlock, self).__init__()
        
        self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
        self.idwt = DWTInverse(mode='zero', wave=wavelet)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.AvgPool2d(kernel_size=2, stride=2) if stride != 1 else nn.Identity()
            )
        
        self.stride = stride

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Apply DWT
        yl, yh = self.dwt(out)
        
        out = self.conv2(yl)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply IDWT
        out = self.idwt((out, yh))
        
 
        out = self.se(out)
        
        if self.stride != 1:
            out = F.avg_pool2d(out, 2)
        if out.shape != residual.shape:
            size = residual.shape[-2], residual.shape[-1]
            out = F.interpolate(out, size=size, mode='bilinear')

        out += residual
        out = self.relu(out)
        
        return out

class WaveletResNet(nn.Module):
    def __init__(self):
        super(WaveletResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 8, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )
        #self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(WaveletCNNBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(WaveletCNNBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        x  = self.layer1(x)

        
        l2_output = self.layer2(x)
        x = self.layer3(l2_output)
        logits = self.avgpool(x)
        logits = torch.flatten(logits,1)
        #print(logits.shape)
        logits = self.mlp(logits)
        return l2_output, logits   #feature, out(logits), inter

# Test the model
# model = WaveletResNet()
# data = torch.randn(2, 3, 400, 544)
# print(model(data).shape)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")
