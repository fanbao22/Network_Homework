import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# 1. 超参数设置
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. SVHN 数据预处理
svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(svhn_mean, svhn_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(svhn_mean, svhn_std)
])


# 3. 构建模型一：CustomCNN
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 4. 构建模型二：ResNet18
class ModifiedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        # 修改第一层卷积以匹配 32x32 小尺寸输入
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)


# 5. 训练与测试单轮函数（优化为传参模式，避免全局变量依赖）
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = outputs.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(dataloader), 100 * correct / total


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / len(dataloader), 100 * correct / total



if __name__ == '__main__':
    # 加载数据集（必须在 main 中加载以支持 Windows 多进程）
    train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=False, transform=transform_train)
    test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 选择模型
    USE_RESNET = True
    if USE_RESNET:
        model = ModifiedResNet18().to(DEVICE)
    else:
        model = CustomCNN().to(DEVICE)

    # 训练准备
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 记录所需指标
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # 构建训练循环
    print("训练开始...")
    for epoch in range(EPOCHS):
        # 显式传入 model, loader 等参数
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        scheduler.step()
        print(f"[{epoch + 1:2d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 绘制可视化结果
    epochs = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(12, 5))

    # 图1：训练损失和测试损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs, test_losses, label="Test Loss", color='red')
    plt.title("Train & Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)

    # 图2：训练准确率和测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc", color='orange')
    plt.plot(epochs, test_accs, label="Test Acc", color='green')
    plt.title("Train & Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("svhn_training_curves_30epochs.png", dpi=300)
    plt.close()

    # 保存模型
    print("训练完成！图片已保存：svhn_training_curves_30epochs.png")
    torch.save(model.state_dict(), "svhn_model.pth")