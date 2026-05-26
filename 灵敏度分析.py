import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from mpl_toolkits.mplot3d import Axes3D

# ---- Model coefficients (assumed to be obtained from regression or analysis) ----
a_V, b_V, c_V, d_V = 0.295, -3.637e-7, 1.243e-13, 6.505e-7
a_C, b_C, c_C = 53.5612, 1.33e13, 2.756e8
a_S, b_S, c_S = 0.1366, -8.949e-14, -0.9032

class MultiObjectiveProblem(Problem):
    def __init__(self, revenue_limit, infrastructure_limit):
        super().__init__(
            n_var=1,
            n_obj=4,
            n_ieq_constr=4,
            xl=np.array([1120324]),
            xu=np.array([1650149])
        )
        self.revenue_limit = revenue_limit
        self.infrastructure_limit = infrastructure_limit

    def _evaluate(self, V, out, *args, **kwargs):
        v = V[:, 0]

        W = 1500 * v
        F = a_V * v + b_V * v**2 + c_V * v**3 + d_V
        I = a_C * v + b_C * (1 / v) + c_C
        S = a_S * np.log(1 + v) + b_S * v**2 + c_S

        out["F"] = np.column_stack([-W, F, I, -S])

        g = np.column_stack([
            F - self.revenue_limit,     # F <= revenue_limit
            I - self.infrastructure_limit,  # I <= infrastructure_limit
            0.738 - S,  # S >= 0.738
            300000000 - I  # I >= 300,000,000
        ])
        
        out["G"] = g

# Constraints limits to analyze
revenue_limits = [60000, 70000, 80000]  # Example: varying revenue limits
infrastructure_limits = [345000000, 500000000, 550000000]  # Example: varying infrastructure limits

# Create a figure with 3D subplots
num_revenue_limits = len(revenue_limits)
num_infra_limits = len(infrastructure_limits)

fig = plt.figure(figsize=(18, 12))

# Prepare to collect results for visualization
results = []

for rev_limit in revenue_limits:
    for infra_limit in infrastructure_limits:
        problem = MultiObjectiveProblem(revenue_limit=rev_limit, infrastructure_limit=infra_limit)

        # Running NSGA-II algorithm
        algorithm = NSGA2(pop_size=100)
        res = minimize(
            problem,
            algorithm,
            ("n_gen", 200),  # Number of generations
            seed=42,
            verbose=False
        )

        # Store results for this specific combination of constraints
        results.append((res.F, rev_limit, infra_limit))

# Plot all Pareto fronts in separate subplots
for i, (pareto_front, rev_limit, infra_limit) in enumerate(results):
    ax = fig.add_subplot(num_revenue_limits, num_infra_limits, i + 1, projection='3d')
    if pareto_front is not None and len(pareto_front) > 0:
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], s=30)
        ax.set_title(f"Rev Lim: {rev_limit}, Infra Lim: {infra_limit}")
        ax.set_xlabel("Revenue (W)")
        ax.set_ylabel("Carbon Emission (F)")
        ax.set_zlabel("Infrastructure Cost (I)")
    else:
        ax.text(0.5, 0.5, 0.5, "No feasible solution found.", fontsize=12, ha='center')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
