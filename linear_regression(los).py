import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: 读取数据并确保列名唯一
file_path = "/Users/fujunhan/Desktop/los 1.xlsx"  # 确保文件路径正确
data = pd.read_excel(file_path)

# Step 2: 检查列名并提取相关列
print(data.columns)  # 打印列名以确保列名唯一

years = data['年份']
griffith_visitors = data['Avila Adobe visitors']

filtered_data = data[(years >= 2014) & (years <= 2023)]

filtered_years = filtered_data['年份']
filtered_Avila_Adobe_visitors = filtered_data['Avila Adobe visitors']

# Step 4: 进行线性回归分析
# 添加常量项
X = sm.add_constant(filtered_years)  # 添加常数项以适应截距
model = sm.OLS(filtered_Avila_Adobe_visitors, X).fit()  # 拟合模型

# 打印回归结果
print(model.summary())

# Step 5: 可视化回归结果
plt.figure(figsize=(10, 5))

# 绘制散点图
plt.scatter(filtered_years, filtered_Avila_Adobe_visitors, color='blue', label='ATIC visitors', alpha=0.5)

# 绘制回归线
plt.plot(filtered_years, model.predict(X), color='red', label='Regression Line')

# 添加标题和标签
plt.title('Linear Regression of Avila Adobe Visitors (2014-2023)')
plt.xlabel('Year')
plt.ylabel('Number of Avila Adobe Visitors')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
# 回归参数
const = 48200000
coeff = -23770

# 预测年份
years = [2009, 2010, 2011, 2012, 2013]

# 计算预测值
predictions = {year: const + coeff * year for year in years}

# 输出预测结果
for year, prediction in predictions.items():
    print(f"Year: {year}, Predicted ATIC Visitors: {prediction:.2f}")
