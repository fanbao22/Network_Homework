import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# 1. 自定义 Dataset 类来读取 .mat 文件
class SVHNDataset(Dataset):
    def __init__(self, file_path, transform=None):
        # 加载 .mat 文件
        data = loadmat(file_path)
        # 调整维度从 (32, 32, 3, N) 变为 (N, 32, 32, 3)
        self.images = np.transpose(data['X'], (3, 0, 1, 2))
        self.labels = data['y'].flatten()

        # 修正标签：将标签 10 映射为 0
        self.labels[self.labels == 10] = 0
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
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

    print(f"本地数据加载成功！训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")


    # 2. 构建 CNN 模型
    class SVHNNet(nn.Module):
        def __init__(self):
            super(SVHNNet, self).__init__()
            self.features = nn.Sequential(
                # 卷积层 1: 输入 3 通道, 输出 16 通道, 卷积核 3x3
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

                # 卷积层 2: 输出 32 通道
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 10)  # 10 个类别 (数字 0-9)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)  # 展平
            x = self.classifier(x)
            return x


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVHNNet().to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVHNNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. 带有指标追踪的训练循环
    epochs = 10
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()

        # 测试阶段 (每个 epoch 评估一次)
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * imgs.size(0)
                _, pred = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (pred == labels).sum().item()

        # 记录数据
        history['train_loss'].append(train_loss / train_total)
        history['test_loss'].append(test_loss / test_total)
        history['train_acc'].append(train_correct / train_total)
        history['test_acc'].append(test_correct / test_total)

        print(f"Epoch {epoch + 1}: Train Acc {history['train_acc'][-1]:.4f}, Test Acc {history['test_acc'][-1]:.4f}")

    # --- 4. 绘制评价指标曲线 ---
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, epochs + 1)

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()