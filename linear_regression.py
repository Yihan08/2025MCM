import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Step 1: 读取数据
file_path = "/Users/fujunhan/Desktop/2025美赛B题/Tables 2.xlsx"  # 文件路径
data = pd.read_excel(file_path)

# Step 2: 重命名列
data.rename(columns={
    "访客数量(Juneau)": "x",
    "政府旅游收入": "y"
}, inplace=True)

# Step 3: 查看数据的前几行
print(data.head())

# Step 4: 进行线性回归
slope, intercept, r_value, p_value, std_err = linregress(data['x'], data['y'])

# 计算拟合值
data['y_fit'] = slope * data['x'] + intercept

# Step 5: 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'], color='blue', label='Observed Data', alpha=0.6)
plt.plot(data['x'], data['y_fit'], color='red', label='Fitted Line', linewidth=2)

# 添加标签和标题
plt.xlabel('Visitor Count (Juneau)', fontsize=12)
plt.ylabel('Government Tourism Revenue', fontsize=12)
plt.title('Linear Regression of Visitor Count and Government Tourism Revenue', fontsize=14)
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

# Step 6: 输出线性回归的参数
print(f"Linear Regression Equation: y = {slope:.4f} * x + {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}, P-value: {p_value:.4f}")