from SVHN_classify import SVHNDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 定义预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 numpy (H, W, C) 转化为 tensor (C, H, W)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 实例化 Dataset
train_dataset = SVHNDataset('train_32x32.mat', transform=transform)
test_dataset = SVHNDataset('test_32x32.mat', transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 2. 模型定义 ---

# 基础 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

# ResNet 残差块与 Tiny-ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x): return torch.relu(self.convs(x) + self.shortcut(x))

class TinyResNet(nn.Module):
    def __init__(self):
        super(TinyResNet, self).__init__()
        self.prep = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = ResidualBlock(32, 64, 2) # 16x16
        self.layer2 = ResidualBlock(64, 128, 2) # 8x8
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 8, 10))
    def forward(self, x): return self.fc(self.layer2(self.layer1(self.prep(x))))

# --- 3. 训练与评估通用函数 ---

def train_model(model, name, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    res = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        t_loss, t_corr, t_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            t_loss += loss.item() * x.size(0)
            t_corr += (out.argmax(1) == y).sum().item(); t_total += y.size(0)

        model.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_corr += (out.argmax(1) == y).sum().item(); v_total += y.size(0)

        res['train_loss'].append(t_loss/t_total); res['test_loss'].append(v_loss/v_total)
        res['train_acc'].append(t_corr/t_total); res['test_acc'].append(v_corr/v_total)
        print(f"[{name}] Epoch {epoch+1} Test Acc: {res['test_acc'][-1]:.4f}")
    return res

# --- 4. 执行对比实验 ---
cnn_history = train_model(SimpleCNN(), "SimpleCNN", epochs=10)
resnet_history = train_model(TinyResNet(), "ResNet", epochs=10)

# --- 5. 绘图对比 ---
plt.figure(figsize=(14, 6))
epochs_range = range(1, 11)

# 准确率对比
plt.subplot(1, 2, 1)
plt.plot(epochs_range, cnn_history['test_acc'], 'b--', label='SimpleCNN Test Acc')
plt.plot(epochs_range, resnet_history['test_acc'], 'b-', label='ResNet Test Acc')
plt.plot(epochs_range, cnn_history['train_acc'], 'g--', label='SimpleCNN Train Acc')
plt.plot(epochs_range, resnet_history['train_acc'], 'g-', label='ResNet Train Acc')
plt.title('Accuracy Comparison')
plt.legend(); plt.grid()

# 损失对比
plt.subplot(1, 2, 2)
plt.plot(epochs_range, cnn_history['test_loss'], 'r--', label='SimpleCNN Test Loss')
plt.plot(epochs_range, resnet_history['test_loss'], 'r-', label='ResNet Test Loss')
plt.title('Loss Comparison')
plt.legend(); plt.grid()
plt.show()