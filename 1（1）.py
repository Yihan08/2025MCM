import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: 读取数据并确保列名唯一
file_path = "/Users/fujunhan/Desktop/2025美赛B题/los.xlsx"  # 确保文件路径正确
data = pd.read_excel(file_path)

# Step 2: 重命名列
data.rename(columns={
    "Chinese American Museum visitors": "x",
    "CAM MT CO2e": "y"
}, inplace=True)

# Step 3: 查看数据的前几行
print(data.head())

# Step 4: 定义自变量和因变量（只保留线性项）
X = data[['x']]  # 只包含一阶项的自变量
X = sm.add_constant(X)  # 添加常数项（截距项）
y = data['y']  # 因变量

# Step 5: 进行线性回归
model = sm.OLS(y, X).fit()  # 创建并拟合模型

# Step 6: 输出回归结果
print(model.summary())

# Step 7: 绘制图形
# 生成预测值
data['y_pred'] = model.predict(X)

# 按 x 值对数据进行排序，以避免折线现象
data_sorted = data.sort_values('x')

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(data_sorted['x'], data_sorted['y'], color='blue', label='Observed Data', alpha=0.7)  # 原始数据点
plt.plot(data_sorted['x'], data_sorted['y_pred'][data_sorted.index], color='red', label='Fitting Model', linewidth=2)  # 拟合直线
plt.title('Linear Fitting Curve between Visitors and CO2 Emissions')
plt.xlabel('Amount of Visitors')
plt.ylabel('MT CO2e')
plt.legend()
plt.grid()
plt.show()
