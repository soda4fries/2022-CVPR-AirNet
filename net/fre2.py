import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
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
    def __init__(self, in_channels, out_channels, wavelet='db1', levels=3):
        super(WaveletCNNBlock, self).__init__()
        self.levels = levels
        
        self.dwt = DWTForward(J=levels, mode='zero', wave=wavelet)
        
        # CNN branch (ResNet branch)
        self.cnn_branch = nn.Sequential(
            SeparableConv2d(in_channels, 32),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            SeparableConv2d(32, 64),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            SeparableConv2d(64, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            SEBlock(out_channels)
        )
        
        # Low frequency branch with 1x1, 3x3, 1x1 convolutions
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            SeparableConv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            SEBlock(in_channels)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            SEBlock(out_channels)
        )
        
        if in_channels != out_channels:
            self.residual_adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual_adjust = None

    def forward(self, x):
        # CNN branch
        cnn_output = self.cnn_branch(x)
        
        # Wavelet decomposition (low frequency only)
        yl, _ = self.dwt(x)
        
        # Low frequency branch
        low_freq = self.low_freq_branch(yl)
        
        # Interpolate low frequency to match CNN output size
        low_freq_upsampled = F.interpolate(low_freq, size=cnn_output.shape[-2:], mode='bilinear', align_corners=False)
        
        # Fusion of CNN and Low frequency branches
        fused = torch.cat([cnn_output, low_freq_upsampled], dim=1)
        output = self.fusion(fused)
        
        # Residual connection
        if self.residual_adjust is not None:
            x = self.residual_adjust(x)
        
        return output + x

class WaveletResNet(nn.Module):
    def __init__(self, num_classes=37):
        super(WaveletResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(WaveletCNNBlock(in_channels, out_channels, levels=2))
        for _ in range(1, blocks):
            layers.append(WaveletCNNBlock(out_channels, out_channels, levels=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        logits = self.avgpool(x)
        logits = torch.flatten(logits,1)
        #print(logits.shape)
        logits = self.mlp(logits)

        return x, logits

# Transformations for CIFAR-100
# transform_train = transforms.Compose([
#     transforms.Pad(4),  # Zero-padding
#     transforms.RandomCrop(32),  # Randomly crop 32x32 from 40x40 padded image
#     transforms.RandomHorizontalFlip(),  # Random horizontal flip
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
# ])

# transform_val = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
# ])

# # Load CIFAR-100 Dataset
# train_dataset = CIFAR100(root='./data', train=True, transform=transform_train, download=True)
# val_dataset = CIFAR100(root='./data', train=False, transform=transform_val, download=True)

# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
# val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16)

# # Define your model here
# model = WaveletResNet(num_classes=100).to(device)  # Adjust num_classes for CIFAR-100

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

# # Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

# Training function
def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# model = WaveletResNet(num_classes=100).to(device)  # Adjust num_classes for CIFAR-100

# inputs = torch.randn(2,3, 128,128)
# outputs = model(inputs)
# print(outputs[0].shape, outputs[1].shape)
# Training loop
# num_epochs = 160
# for epoch in range(num_epochs):
#     train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device)
#     val_loss, val_acc = validate(model, val_loader, criterion, device)
    
#     # Update learning rate
#     scheduler.step()
    
#     print(f"Epoch {epoch+1}/{num_epochs}")
#     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#     print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
#     print()

# print("Training completed!")