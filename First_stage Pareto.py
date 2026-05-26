import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---- 基础模型定义 ----
# Model coefficients (assumed from regression)
a_V, b_V, c_V, d_V = 0.295, -3.637e-7, 1.243e-13, 6.505e-7
a_C, b_C, c_C = 53.5612, 1.33e13, 2.756e8
a_s, b_s, c_s, d_s = -12.337, -3.921e-12, 155.198, 1.992e-5

class MultiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=1,  # One decision variable (number of visitors, V)
            n_obj=4,  # Four objectives
            n_ieq_constr=4,  # Four constraints
            xl=np.array([1120324]),  # Lower bound of visitors
            xu=np.array([1650149])  # Upper bound of visitors
        )

    def _evaluate(self, V, out, *args, **kwargs):
        v = V[:, 0]  # Extract the decision variable

        # Objective functions
        W = 52 * V + 76502404  # Revenue
        F = a_V * v + b_V * v**2 + c_V * v**3 + d_V  # Carbon Emission
        I = a_C * v + b_C * (1 / v) + c_C  # Infrastructure Cost
        S = a_s * np.log(1 + v) + b_s * v**2 + c_s + d_s * v # Satisfaction

        # Objectives array: [-W, F, I, -S]
        out["F"] = np.column_stack([-W, F, I, -S])

        # Constraints array: g(x) <= 0
        g = np.column_stack([
            F - 60000,              # F <= 60000
            I - 450000000,          # I <= 450000,000
            0.738 - S,              # S >= 0.738
            300000000 - I           # I >= 300,000,000
        ])
        out["G"] = g

# ---- NSGA-II optimization ----
algorithm = NSGA2(pop_size=100)
problem = MultiObjectiveProblem()
res = minimize(
    problem,
    algorithm,
    ("n_gen", 200),  # Number of generations
    seed=42,
    verbose=True
)

# ---- CRITIC method to determine weights ----
def critic_weight_calculation(X):
    """
    X: 2D array of objective values (each column is an objective)
    Returns: weights array (importance of each objective)
    """
    # Step 1: Normalize the decision matrix (Min-Max Scaling)
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # Step 2: Compute standard deviations (variability)
    std_devs = np.std(X_norm, axis=0)

    # Step 3: Compute correlation matrix
    correlation_matrix = np.corrcoef(X_norm, rowvar=False)
    conflict_intensity = 1 - correlation_matrix

    # Step 4: Compute the CRITIC value for each objective
    C = std_devs * np.sum(conflict_intensity, axis=1)

    # Step 5: Normalize the weights
    weights = C / np.sum(C)
    return weights

# Extract Pareto Front solutions (res.F)
objectives = res.F[:,:]  # Ignore revenue (W) for weight calculation
weights = critic_weight_calculation(objectives)
print(f"CRITIC Weights for Objectives (W, F, I, S): {weights}")

# ---- TOPSIS method for optimal solution ranking ----
def topsis_ranking(X, weights):
    """
    X: 2D array of objective values
    weights: weights for each objective
    Returns: Index of the best solution
    """
    # Normalize the decision matrix
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # Weighted normalized matrix
    X_weighted = X_norm * weights

    # Ideal (best) and anti-ideal (worst) solutions
    ideal_solution = np.max(X_weighted, axis=0)
    anti_ideal_solution = np.min(X_weighted, axis=0)

    # Euclidean distances to ideal and anti-ideal
    dist_to_ideal = np.linalg.norm(X_weighted - ideal_solution, axis=1)
    dist_to_anti_ideal = np.linalg.norm(X_weighted - anti_ideal_solution, axis=1)

    # TOPSIS score (relative closeness to ideal solution)
    scores = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)

    # Return the index of the best solution
    return np.argmax(scores), scores

# Perform TOPSIS
best_index, scores = topsis_ranking(objectives, weights)
print(f"Best solution index: {best_index}")
print(f"TOPSIS Scores: {scores}")

# ---- Redistribute additional income based on weights ----
E = 10000000  # Additional income (e.g., $10M)
W_share = weights[0] * E
F_share = weights[1] * E
I_share = weights[2] * E
S_share = weights[3] * E

print(f"Additional income allocation:")
print(f"  Revenue (W): {W_share:.2f}")
print(f"  Carbon Emission Reduction (F): {F_share:.2f}")
print(f"  Infrastructure Investment (I): {I_share:.2f}")
print(f"  Visitor Satisfaction (S): {S_share:.2f}")

# ---- Visualize Pareto front solutions ----
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d") if res.F.shape[1] >= 3 else fig.add_subplot()

ax.scatter(res.F[:, 0], res.F[:, 1], res.F[:, 2] if res.F.shape[1] >= 3 else None, 
           c='red', label="Pareto Front", s=30)

ax.scatter(res.F[best_index, 0], res.F[best_index, 1], res.F[best_index, 2] if res.F.shape[1] >= 3 else None, 
           c='blue', label="Best Solution", s=100, edgecolor='black', marker='o')

ax.set_title("Pareto Front without Extra Revenue")
ax.set_xlabel("Objective 1 (W)")
ax.set_ylabel("Objective 2 (F)")
if res.F.shape[1] >= 3:
    ax.set_zlabel("Objective 3 (I)")
ax.legend()
plt.show()

# ---- Detailed output of the best solution ----
optimal_V = res.X[best_index][0]
optimal_W = 52 * optimal_V + 76502404
optimal_F = a_V * optimal_V + b_V * optimal_V**2 + c_V * optimal_V**3 + d_V
optimal_I = a_C * optimal_V + b_C * (1 / optimal_V) + c_C
optimal_S = a_s * np.log(1 + optimal_V) + b_s * optimal_V**2 + c_s + d_s * optimal_V

print(f"Optimal Solution Details:")
print(f"  Number of Visitors (V): {optimal_V:.0f}")
print(f"  Revenue (W): {optimal_W:.2f}")
print(f"  Carbon Emission (F): {optimal_F:.2f}")
print(f"  Infrastructure Cost (I): {optimal_I:.2f}")
print(f"  Satisfaction (S): {optimal_S:.2f}")
