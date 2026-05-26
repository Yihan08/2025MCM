import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 读取数据并确保列名唯一
file_path = "/Users/fujunhan/Desktop/Book1.xlsx"  # 确保文件路径正确
data = pd.read_excel(file_path)

# Step 2: 重命名列
data.rename(columns={
    "positive_impact": "y",
    "visitors":  "x"
}, inplace=True)

# Step 3: 查看数据的前几行
print(data.head())

# Step 4: 数据转换
data['ln(1+x)'] = np.log(1 + data['x'])
data['x^2'] = data['x']**2  # 添加 x 的平方项

# Step 5: 定义自变量和因变量
X = data[['ln(1+x)', 'x^2']]  # 自变量包括 ln(1+x) 和 x 的平方
X = sm.add_constant(X)  # 添加常数项（截距项）
y = data['y']  # 因变量

# Step 6: 进行线性回归
model = sm.OLS(y, X).fit()  # 创建并拟合模型

# Step 7: 输出回归结果
print(model.summary())

# Step 8: 绘制图形
# 生成预测值
data['y_pred'] = model.predict(X)

# 生成平滑曲线
x_values = np.linspace(data['x'].min(), data['x'].max(), 100)  # 创建 100 个细分的 x 值
ln_1_x_values = np.log(1 + x_values)  # 计算 ln(1+x)
x2_values = x_values**2  # 计算 x 的平方

# 创建用于预测的新 DataFrame
X_smooth = pd.DataFrame({
    'const': np.ones_like(x_values),  # 常数项
    'ln(1+x)': ln_1_x_values,
    'x^2': x2_values
})

# 计算平滑的 y 值
y_pred_smooth = model.predict(X_smooth)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'], color='blue', label='实际值', alpha=0.7)  # 原始数据点
plt.plot(x_values, y_pred_smooth, color='red', label='拟合曲线', linewidth=2)  # 拟合曲线
plt.title('y~x')
plt.xlabel('visitors')
plt.ylabel('positive_impact')
plt.legend()
plt.grid()
plt.show()
