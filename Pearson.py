import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

# Glacier data
x_glacier = np.array([1948, 1975, 1990, 2005, 2015, 2019])  
y_glacier = np.array([4755.5, 4584.0, 4496.1, 4238.7, 4001.5, 3816.3]) 

# Load CO2 data
file_path = "/Users/fujunhan/Desktop/2025美赛B题/Tables 2.xlsx" 
data = pd.read_excel(file_path)
data.rename(columns={"年份": "Year", "MT CO2e": "CO2"}, inplace=True)
x_co2 = data["Year"].values 
y_co2 = data["CO2"].values  

# Interpolation
interp_years = np.union1d(x_glacier, x_co2)
interp_years = interp_years[interp_years >= 2000]

glacier_interp_func = interp1d(x_glacier, y_glacier, kind='quadratic', fill_value='extrapolate') 
y_glacier_interp = glacier_interp_func(interp_years) 

co2_interp_func = interp1d(x_co2, y_co2, kind='linear', fill_value='extrapolate') 
y_co2_interp = co2_interp_func(interp_years)  

# Polynomial fitting
degree_glacier = 2
coeff_glacier = np.polyfit(interp_years, y_glacier_interp, degree_glacier)
y_pred_glacier = np.polyval(coeff_glacier, interp_years)

degree_co2 = 1 
coeff_co2 = np.polyfit(interp_years, y_co2_interp, degree_co2)
y_pred_co2 = np.polyval(coeff_co2, interp_years)

# Pearson correlation coefficient
pearson_corr, p_value = pearsonr(y_pred_glacier, y_pred_co2)

# Print Pearson correlation results
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"P-value: {p_value:.4f}")

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Glacier area plot
ax1.scatter(interp_years, y_glacier_interp, color='slateblue', label='Interpolated Glacier Area')
ax1.plot(interp_years, y_pred_glacier, color='slateblue', label=f'{degree_glacier}-Degree Polynomial Fit (Glacier)')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Total Glacier Area (km²)', color='black', fontsize=12)
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')

# CO2 emissions plot
ax2 = ax1.twinx()
ax2.scatter(interp_years, y_co2_interp, color='thistle', label='Interpolated CO2 Emissions')
ax2.plot(interp_years, y_pred_co2, color='thistle', linestyle='--', label=f'{degree_co2}-Degree Polynomial Fit (CO2)')
ax2.set_ylabel('CO2 Emissions (MT)', color='black', fontsize=12)
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(loc='upper right')

# Add Pearson correlation coefficient to the plot
plt.text(0.5, 0.9, f'Pearson Correlation Coefficient: {pearson_corr:.4f}\nP-value: {p_value:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, ha='center')

# Title and grid
plt.title('Regression Analysis of Glacier Area and CO2 Emissions (Interpolated Years)', fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()

