import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 数据准备
df = pd.read_csv('Concrete_Data_Yeh.csv')
X = df.drop(columns=['csMPa']).values
y = df['csMPa'].values

# 按照作业要求：将数据集的前 80% 作为训练集，后 20% 作为测试集
train_size = int(0.8 * len(df))
X_train_raw, X_test_raw = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 2. 数据预处理
# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 主成分分析 (PCA) 降维
# 提取能够解释大部分数据方差（95%）的主成分
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

n_components = X_train_pca.shape[1]
print(f"原始特征维度: 8, PCA降维后维度: {n_components}")

# 3. 线性回归模型构建与训练
lr_model = LinearRegression()
lr_model.fit(X_train_pca, y_train)

# 4. 模型测试和评估
y_pred = lr_model.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred)
print(f"测试集均方误差 (Test MSE): {mse:.4f}")

# 5. 可视化性能评估
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual Strength (MPa)')
plt.ylabel('Predicted Strength (MPa)')
plt.title(f'Linear Regression: Actual vs Predicted (PCA: {n_components})')
plt.legend()
plt.grid(True)
plt.show()