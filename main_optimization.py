import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
from sklearn.preprocessing import MinMaxScaler

# ---- Model coefficients (assumed to be obtained from regression or analysis) ----
a_V, b_V, c_V, d_V = 0.295, -3.637e-7, 1.243e-13, 6.505e-7
a_C, b_C, c_C = 53.5612, 1.33e13, 2.756e8
a_s, b_s, c_s, d_s = -12.337, -3.921e-12, 155.198, 1.992e-5

class MultiObjectiveProblem(Problem):
    def __init__(self):
        # n_var=1: Only one decision variable (number of visitors, V)
        # n_obj=4: Four objective functions
        # n_ieq_constr=4: Four inequality constraints
        # xl and xu define lower/upper bounds for V
        super().__init__(
            n_var=1,
            n_obj=4,
            n_ieq_constr=4,
            xl=np.array([1120324]),
            xu=np.array([1650149])
        )

    def _evaluate(self, V, out, *args, **kwargs):
        # Extract the decision variable from the array
        v = V[:, 0]

        # Calculate the four objectives:
        # 1) Revenue (W) - must be maximized => use negative sign for minimization procedure
        W = 52 * v + 76502404
        # 2) Carbon Emission (F) - to be minimized
        F = a_V * v + b_V * v**2 + c_V * v**3 + d_V
        # 3) Infrastructure Cost (I) - to be minimized
        I = a_C * v + b_C * (1 / v) + c_C
        # 4) Satisfaction (S) - must be maximized => use negative sign for minimization
        S = a_s * np.log(1 + v) + b_s * v**2 + c_s + d_s * v

        # Collect objectives in a 2D array: [-W, F, I, -S]
        out["F"] = np.column_stack([-W, F, I, -S])

        # Define inequality constraints: g(x) <= 0
        g = np.column_stack([
            F - 60000,              # F <= 60000
            I - 450000000,          # I <= 450,000,000
            0.738 - S,              # S >= 0.738
            300000000 - I           # I >= 300,000,000
        ])
        out["G"] = g

# ---- NSGA-II algorithm for multi-objective optimization ----
algorithm = NSGA2(pop_size=100)

# ---- Solve the multi-objective problem ----
problem = MultiObjectiveProblem()
res = minimize(
    problem,
    algorithm,
    ("n_gen", 200),  # Number of generations for the optimization
    seed=42,
    verbose=True
)

# ---- Visualize the Pareto front (only first 3 objectives in scatter plot) ----
if res.F is not None and len(res.F) > 0:
    plot = Scatter(title="Pareto Front (First 3 Objectives)")
    plot.add(res.F[:, :3], s=30, facecolors='r', edgecolors='r')
    plot.show()
else:
    print("Feasible solution not found.")

# ---- Print the Pareto solutions found by the algorithm ----
for i, sol in enumerate(res.X):
    optimal_V = sol[0]

    # Recalculate each objective for clarity
    optimal_W = 52 * optimal_V  + 76502404
    optimal_F = a_V * optimal_V + b_V * (optimal_V**2) + c_V * (optimal_V**3) + d_V
    optimal_I = a_C * optimal_V + b_C * (1 / optimal_V) + c_C
    optimal_S = a_s * np.log(1 + optimal_V) + b_s * (optimal_V**2) + c_s + d_s * optimal_V

    print(f"Solution {i + 1}:")
    print(f"  Number of Visitors (V): {optimal_V:.0f}")
    print(f"  Revenue (W): {optimal_W:.2f}")
    print(f"  Carbon Emission (F): {optimal_F:.2f}")
    print(f"  Infrastructure Cost (I): {optimal_I:.2f}")
    print(f"  Satisfaction (S): {optimal_S:.2f}")

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