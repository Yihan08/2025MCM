import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
from sklearn.preprocessing import MinMaxScaler

# ---- Model coefficients (assumed to be obtained from regression or analysis) ----
fh0, fh1 = 1.932e5, 0.0033
fd0, fd1 = -5381, 0.0268
fg0, fg1, fg2, fg3 = 1.444e-7, 0.0978, -3.064e-8, -1.64e-15
ft0, ft1 = 7338.16, -60581.25
fa0, fa1 = 11989.43, 0.166
fc0, fc1 = 159.314, 0.1693

c1, c2, c3, c4, c5, c6 = 300, 400, 150, 100, 100, 100
vopt1, vopt2, vopt3, vopt4, vopt5, vopt6 = 18250000, 31025000, 1095000, 365000, 500000, 100000
si0, si1, si2 = 0.54329, -1.59e-8, -1.286e-15
sj0, sj1, sj2 = 0.54329, -1.59e-8, -1.286e-15
ii, ij = 30.688, 30.688

# ---- MultiObjectiveProblem class ----
class MultiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=6,
            n_obj=4,
            n_ieq_constr=11,
            xl=np.array([0, 0, 0, 120000, 374760, 37284]),
            xu=np.array([9660000, 18666000, 2382600, 365000, 500000, 100000])
        )

    def _evaluate(self, X, out, *args, **kwargs):
        v1, v2, v3, v4, v5, v6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]

        # Objective functions
        W = v1 * c1 + v2 * c2 + v3 * c3 + v4 * c4 + v5 * c5 + v6 * c6
        f1 = fh0 + fh1 * v1
        f2 = fd0 + fd1 * v2
        f3 = fg0 + fg1 * v3 + fg2 * v3**2 + fg3 * v3**3
        f4 = ft0 * np.log(v4) + ft1
        f5 = fa0 + fa1 * v5
        f6 = fc0 + fc1 * v6
        F = f1 + f2 + f3 + f4 + f5 + f6
        I = ii * (v1 + v2 + v3) + ij * (v4 + v5 + v6)
        S = (si0 + si1 * (v1-vopt1) + si2 * (v1-vopt1)**2 +
             si0 + si1 * (v2-vopt2) + si2 * (v2-vopt2)**2 +
             si0 + si1 * (v3-vopt3) + si2 * (v3-vopt3)**2 +
             sj0 + sj1 * (v4-vopt4) + sj2 * (v4-vopt4)**2 +
             sj0 + sj1 * (v5-vopt5) + sj2 * (v5-vopt5)**2 +
             sj0 + sj1 * (v6-vopt6) + sj2 * (v6-vopt6)**2) / 6

        out["F"] = np.column_stack([-W, F, I, -S])

        # Constraints
        g = np.column_stack([
            fh0 + fh1 * v1 - 225529,
            fd0 + fd1 * v2 - 1542174,
            fg0 + fg1 * v3 + fg2 * v3**2 + fg3 * v3**3 - 85932,
            30000 - (ft0 * np.log(v4) + ft1),
            80000 - (fa0 + fa1 * v5),
            8100 - (fc0 + fc1 * v6),
            v1 + v2 + v3 + v4 + v5 + v6 - 32000000,
            0.53 - S,
            I - 2000000000,
            900000000 - I,
            0 - F
        ])
        
        out["G"] = g

# ---- NSGA-II optimization ----
algorithm = NSGA2(pop_size=100)
problem = MultiObjectiveProblem()
res = minimize(
    problem,
    algorithm,
    ("n_gen", 200),
    seed=42,
    verbose=True
)

if not (res.F is not None and len(res.F) > 0):
    print("No solution.")    

# ---- CRITIC method for dynamic weight calculation ----
def critic_weight_calculation(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    std_devs = np.std(X_norm, axis=0)
    correlation_matrix = np.corrcoef(X_norm, rowvar=False)
    conflict_intensity = 1 - correlation_matrix
    C = std_devs * np.sum(conflict_intensity, axis=1)
    weights = C / np.sum(C)
    return weights

# Extract Pareto Front and apply CRITIC
objectives = res.F[:, 1:]  # Ignore revenue (W) for weight calculation
weights = critic_weight_calculation(objectives)
print(f"CRITIC Weights for Objectives (F, I, S): {weights}")

# ---- TOPSIS method ----
def topsis_ranking(X, weights):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    X_weighted = X_norm * weights
    ideal_solution = np.max(X_weighted, axis=0)
    anti_ideal_solution = np.min(X_weighted, axis=0)
    dist_to_ideal = np.linalg.norm(X_weighted - ideal_solution, axis=1)
    dist_to_anti_ideal = np.linalg.norm(X_weighted - anti_ideal_solution, axis=1)
    scores = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)
    return np.argmax(scores), scores

# Perform TOPSIS
best_index, scores = topsis_ranking(objectives, weights)
print(f"Best solution index: {best_index}")
print(f"TOPSIS Scores: {scores}")

# ---- Visualize the Pareto front ----
if res.F is not None and len(res.F) > 0:
    plot = Scatter(title="Pareto Front (First 3 Objectives)")
    plot.add(res.F[:, :3], s=30, facecolors='r', edgecolors='r')
    plot.show()
else:
    print("Feasible solution not found.")

# ---- Output optimal solution ----
optimal_solution = res.X[best_index]
optimal_v1, optimal_v2, optimal_v3, optimal_v4, optimal_v5, optimal_v6 = optimal_solution

optimal_W = optimal_v1 * c1 + optimal_v2 * c2 + optimal_v3 * c3 + optimal_v4 * c4 + optimal_v5 * c5 + optimal_v6 * c6
optimal_F = sum([
    fh0 + fh1 * optimal_v1,
    fd0 + fd1 * optimal_v2,
    fg0 + fg1 * optimal_v3 + fg2 * optimal_v3**2 + fg3 * optimal_v3**3,
    ft0 * np.log(optimal_v4) + ft1,
    fa0 + fa1 * optimal_v5,
    fc0 + fc1 * optimal_v6
])
optimal_I = ii * (optimal_v1 + optimal_v2 + optimal_v3) + ij * (optimal_v4 + optimal_v5 + optimal_v6)
optimal_S = (si0 + si1 * (optimal_v1-vopt1) + si2 * (optimal_v1-vopt1)**2 +
             si0 + si1 * (optimal_v2-vopt2) + si2 * (optimal_v2-vopt2)**2 +
             si0 + si1 * (optimal_v3-vopt3) + si2 * (optimal_v3-vopt3)**2 +
             sj0 + sj1 * (optimal_v4-vopt4) + sj2 * (optimal_v4-vopt4)**2 +
             sj0 + sj1 * (optimal_v5-vopt5) + sj2 * (optimal_v5-vopt5)**2 +
             sj0 + sj1 * (optimal_v6-vopt6) + sj2 * (optimal_v6-vopt6)**2) / 6

print(f"Optimal Solution:")
print(f"  V1: {optimal_v1:.0f}")
print(f"  V2: {optimal_v2:.0f}")
print(f"  V3: {optimal_v3:.0f}")
print(f"  V4: {optimal_v4:.0f}")
print(f"  V5: {optimal_v5:.0f}")
print(f"  V6: {optimal_v6:.0f}")
print(f"  Revenue (W): {optimal_W:.2f}")
print(f"  Carbon Emission (F): {optimal_F:.2f}")
print(f"  Infrastructure Cost (I): {optimal_I:.2f}")
print(f"  Satisfaction (S): {optimal_S:.2f}")

