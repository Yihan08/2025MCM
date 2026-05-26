import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: 读取数据并确保列名唯一
file_path = "/Users/fujunhan/Desktop/Tables 2.xlsx"  # 确保文件路径正确
data = pd.read_excel(file_path)

# Step 2: 重命名列
data.rename(columns={
    "访客数量(Juneau)": "x1",
    "当地人口收入": "x2",
    "基础设施维护成本": "y"
}, inplace=True)

# Step 3: 查看数据的前几行
print(data.head())

# Step 4: 定义自变量和因变量
X = data[['x1', 'x2']]  # 自变量
X = sm.add_constant(X)  # 添加常数项（截距项）
y = data['y']  # 因变量

# Step 5: 进行线性回归
model = sm.OLS(y, X).fit()  # 创建并拟合模型

# Step 6: 输出回归结果
print(model.summary())

# 预测值
data['y_pred'] = model.predict(X)

# Step 7: 绘制散点图和回归表面
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制实际数据点
ax.scatter(data['x1'], data['x2'], data['y'], color='blue', label='实际值', alpha=0.7)

# 创建网格以绘制回归面
x1_range = np.linspace(data['x1'].min(), data['x1'].max(), 30)
x2_range = np.linspace(data['x2'].min(), data['x2'].max(), 30)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# 计算预测值
X_grid = sm.add_constant(np.c_[x1_grid.ravel(), x2_grid.ravel()])  # 添加常数项
y_pred_grid = model.predict(X_grid).reshape(x1_grid.shape)  # 预测值调整成网格形状

# 绘制回归面
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='red', alpha=0.5, label='拟合曲线')

# 设置标签
ax.set_title('基础设施维护成本拟合曲面')
ax.set_xlabel('访客数量 (x1)')
ax.set_ylabel('当地人口收入 (x2)')
ax.set_zlabel('基础设施维护成本 (y)')
ax.legend()

plt.show()
