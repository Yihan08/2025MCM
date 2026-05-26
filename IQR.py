import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: 读取数据并确保列名唯一
file_path = "/Users/fujunhan/Desktop/los.xlsx"  # 确保文件路径正确
data = pd.read_excel(file_path)

# Step 2: 检查列名并进行IQR分析
print(data.columns)  # 打印列名以确保列名唯一

# 选择要分析的列
hollywood_visitors = data['Hollywood visitors']
disney_visitors = data['Disney visitors']

def calculate_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    return Q1, Q3, IQR

# 计算 IQR
hollywood_Q1, hollywood_Q3, hollywood_IQR = calculate_iqr(hollywood_visitors)
disney_Q1, disney_Q3, disney_IQR = calculate_iqr(disney_visitors)

print(f"Hollywood Visitors - Q1: {hollywood_Q1}, Q3: {hollywood_Q3}, IQR: {hollywood_IQR}")
print(f"Disney Visitors - Q1: {disney_Q1}, Q3: {disney_Q3}, IQR: {disney_IQR}")

# Step 3: 数据可视化 - 箱线图
plt.figure(figsize=(10, 5))

# 创建箱线图
plt.boxplot([hollywood_visitors, disney_visitors], labels=['Hollywood Visitors', 'Disney Visitors'])

# 添加每个点
plt.plot(np.random.normal(1, 0.04, size=len(hollywood_visitors)), hollywood_visitors, 'o', color='blue', alpha=0.5, label='Hollywood Visitors')
plt.plot(np.random.normal(2, 0.04, size=len(disney_visitors)), disney_visitors, 'o', color='orange', alpha=0.5, label='Disney Visitors')

# 添加标题和标签
plt.title('Distribution of Visitors to Hollywood and Disney')
plt.ylabel('Number of Visitors')
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
