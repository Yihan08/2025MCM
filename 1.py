import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: 读取数据
file_path = "/Users/fujunhan/Desktop/2025美赛B题/los.xlsx"  # 文件路径
data = pd.read_excel(file_path)

# Step 2: 重命名列
data.rename(columns={
    "Chinese American Museum visitors": "x",
    "CAM MT CO2e": "y"
}, inplace=True)

# Step 3: 查看数据的前几行
print(data.head())

# Step 4: 数据归一化（如果 x 和 y 的值差距较大，可归一化）
x_min, x_max = data['x'].min(), data['x'].max()
y_min, y_max = data['y'].min(), data['y'].max()

data['x_norm'] = (data['x'] - x_min) / (x_max - x_min)
data['y_norm'] = (data['y'] - y_min) / (y_max - y_min)

# Step 5: 定义反正切函数模型
def arctan_model(x, a, b, c, d):
    """
    反正切函数模型: y = a * arctan(b * x + c) + d
    """
    return a * np.arctan(b * x + c) + d

# Step 6: 进行非线性拟合
initial_guess = [1, 1, 0, 0]  # 初始猜测参数 [a, b, c, d]

# 用 curve_fit 进行拟合（注意使用归一化数据）
popt, pcov = curve_fit(arctan_model, data['x_norm'], data['y_norm'], p0=initial_guess)

# 输出拟合的参数
a, b, c, d = popt
print(f"Fitted Parameters:a={a}, b={b}, c={c}, d={d}")

# Step 7: 生成归一化预测数据
data['y_pred_norm'] = arctan_model(data['x_norm'], *popt)

# 将预测值还原到原始尺度
data['y_pred'] = data['y_pred_norm'] * (y_max - y_min) + y_min

# Step 8: 绘制图形
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'], color='blue', label='Observed Data', alpha=0.7)  # 原始数据点
plt.plot(data['x'], data['y_pred'], color='red', label='Arctan Fitting Curve', linewidth=2)  # 拟合曲线
plt.title('Improved Arctan Fitting Curve')
plt.xlabel('Amount of Visitors')
plt.ylabel('MT CO2e')
plt.legend()
plt.grid()
plt.show()
