import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 数据准备
df = pd.read_csv('Concrete_Data_Yeh.csv')
X = df.drop(columns=['csMPa']).values
y = df['csMPa'].values.reshape(-1, 1)

# 将数据集的前 80% 作为训练集，后 20% 作为测试集
train_size = int(0.8 * len(df))
X_train_raw, X_test_raw = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 2. 数据预处理
# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 主成分分析 (PCA) 降维
# 提取能够解释大部分数据方差（如 95%）的主成分作为主要特征
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

n_components = X_train_pca.shape[1]
print(f"原始特征维度: 8, PCA降维后维度: {n_components}")

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 3. 神经网络构建
class ConcreteRegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(ConcreteRegressionNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = ConcreteRegressionNet(n_components)
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 模型训练
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 模型测试和评估
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_mse = criterion(predictions, y_test_tensor)
    print(f'测试集均方误差 (Test MSE): {test_mse.item():.4f}')

# 绘图可视化：测试集的 target 与神经网络模型输出的 output
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions.numpy(), alpha=0.6, color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual Strength (MPa)')
plt.ylabel('Predicted Strength (MPa)')
plt.title(f'Neural Network Regression (with PCA: {n_components} components)')
plt.legend()
plt.grid(True)
plt.show()