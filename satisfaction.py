import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---- 1. 加载数据 ----
# 你的 Excel 文件路径
file_path = "Desktop/2025美赛B题/los.xlsx"  # 请替换成你的实际文件路径
data = pd.read_excel(file_path)

# ---- 2. 数据按满意度从大到小排序 ----
data_sorted = data.sort_values(by="Satisfaction", ascending=False)  # 按 'Satisfaction' 降序排列
print("排序后的数据（前 5 行）：")
print(data_sorted.head())  # 打印前 5 行，检验排序是否正确

# 重新赋值
Vi = data_sorted['total_visitors'].values  # 自变量：游客数量
S_res = data_sorted['Satisfaction'].values  # 因变量：满意度

# ---- 3. 定义二阶抛物线模型 ----
def quadratic_model(Vi, a, b, c):
    C_opt = 50000000  # 固定最佳游客数量
    return a * (Vi - C_opt)**2 + b * (Vi - C_opt) + c

# ---- 4. 初始参数 ----
initial_guess = [1e-10, 1e-5, 0.5]  # 初始参数 a, b, c

# ---- 5. 使用 curve_fit 拟合模型 ----
popt, pcov = curve_fit(quadratic_model, Vi, S_res, p0=initial_guess)
a_opt, b_opt, c_opt = popt
print("拟合参数：")
print(f"a = {a_opt:.8e}, b = {b_opt:.8e}, c = {c_opt:.8e}")

# ---- 6. 生成预测曲线 ----
Vi_pred = np.linspace(min(Vi), max(Vi), 500)  # 在自变量范围内生成预测值
S_res_pred = quadratic_model(Vi_pred, a_opt, b_opt, c_opt)

# ---- 7. 绘制图形 ----
plt.figure(figsize=(10, 6))
plt.scatter(Vi, S_res, label="Observed Data (Sorted)", color="blue", s=50)  # 排序后的原始数据
plt.plot(Vi_pred, S_res_pred, label="Quadratic Fitting", color="red", linewidth=2)  # 拟合曲线
plt.axvline(50000000, color="green", linestyle="--", label="C_opt = 50000000")  # C_opt
plt.xlabel("Total Visitors", fontsize=12)
plt.ylabel("Satisfaction", fontsize=12)
plt.title("Quadratic Model Fitting (Sorted Data)", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()
