import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter

# Base Model coefficients
fh0, fh1 = 1.932e5, 0.0033
fd0, fd1 = -5381, 0.0268
fg0, fg1, fg2, fg3 = 1.444e-7, 0.0978, -3.064e-8, -1.64e-15
ft0, ft1 = 7338.16, -60581.25
fa0, fa1 = 11989.43, 0.166
fc0, fc1 = 159.314, 0.1693

# Parameters to be varied in Monte Carlo simulation
base_params = {
    'c1': 300, 
    'c2': 400,
    'c3': 150,
    'c4': 100,
    'c5': 100,
    'c6': 100,
    'ii': 30.688,
    'ij': 30.688
}

# Fixed parameters
vopt1, vopt2, vopt3 = 18250000, 31025000, 1095000
vopt4, vopt5, vopt6 = 365000, 500000, 100000
si0, si1, si2 = 0.54329, -1.59e-8, -1.286e-15
sj0, sj1, sj2 = 0.54329, -1.59e-8, -1.286e-15

class MultiObjectiveProblem(Problem):
    def __init__(self, params):
        # Initialize problem with parameters that can vary
        super().__init__(
            n_var=6,
            n_obj=4,
            n_ieq_constr=10,
            xl=np.array([0, 0, 0, 120000, 374760, 37284]),
            xu=np.array([9660000, 18666000, 2382600, 365000, 500000, 100000])
        )
        # Store varied parameters
        self.c1 = params['c1']
        self.c2 = params['c2']
        self.c3 = params['c3']
        self.c4 = params['c4']
        self.c5 = params['c5']
        self.c6 = params['c6']
        self.ii = params['ii']
        self.ij = params['ij']

    def _evaluate(self, X, out, *args, **kwargs):
        # Extract decision variables
        v1, v2, v3, v4, v5, v6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]

        # Calculate objectives:
        # 1) Revenue (W) - maximize (use negative for minimization)
        W = v1 * self.c1 + v2 * self.c2 + v3 * self.c3 + v4 * self.c4 + v5 * self.c5 + v6 * self.c6
        
        # 2) Carbon Emission (F) - minimize
        f1 = fh0 + fh1 * v1
        f2 = fd0 + fd1 * v2
        f3 = fg0 + fg1 * v3 + fg2 * v3**2 + fg3 * v3**3
        f4 = ft0 * np.log(v4) + ft1
        f5 = fa0 + fa1 * v5
        f6 = fc0 + fc1 * v6
        F = f1 + f2 + f3 + f4 + f5 + f6
        
        # 3) Infrastructure Cost (I) - minimize
        I = self.ii * (v1 + v2 + v3) + self.ij * (v4 + v5 + v6)
        
        # 4) Satisfaction (S) - maximize (use negative for minimization)
        S = (si0 + si1 * (v1-vopt1) + si2 * (v1-vopt1)**2 +
             si0 + si1 * (v2-vopt2) + si2 * (v2-vopt2)**2 +
             si0 + si1 * (v3-vopt3) + si2 * (v3-vopt3)**2 +
             sj0 + sj1 * (v4-vopt4) + sj2 * (v4-vopt4)**2 +
             sj0 + sj1 * (v5-vopt5) + sj2 * (v5-vopt5)**2 +
             sj0 + sj1 * (v6-vopt6) + sj2 * (v6-vopt6)**2) / 6

        out["F"] = np.column_stack([-W, F, I, -S])

        # Define inequality constraints: g(x) <= 0
        g = np.column_stack([
            f1 - 225529,
            f2 - 1542174,
            f3 - 85932,
            30000 - f4,
            80000 - f5,
            8100 - f6,
            v1 + v2 + v3 + v4 + v5 + v6 - 32000000,
            0.53 - S,
            I - 2000000000,
            900000000 - I
        ])
        
        out["G"] = g

def run_optimization(params):
    """Run NSGA-II optimization with given parameters"""
    problem = MultiObjectiveProblem(params)
    algorithm = NSGA2(pop_size=100)
    res = minimize(problem, algorithm, ("n_gen", 200), seed=42, verbose=False)
    return res

def monte_carlo_simulation(base_params, n_samples=50, variation=0.2):
    """
    Monte Carlo simulation with parameter variations
    """
    best_score = None
    best_params = None
    
    # Add tracking for objective ranges
    w_values = []
    f_values = []
    i_values = []
    s_values = []

    def aggregate_obj(values):
        coefficients = np.array([0.3372/11112318100, 0.2091/1976508, 0.2882/1956484900, 0.1655/1])
        return np.sum(values * coefficients)

    for i in range(n_samples):
        print(f"Running Monte Carlo iteration {i+1}/{n_samples}")
        test_params = {}
        for k, base_val in base_params.items():
            if k in ['c1', 'c2', 'c3']:
                low = base_val
                high = base_val * (1 + variation)
            elif k in ['c4', 'c5', 'c6']:
                low = base_val * (1 - variation)
                high = base_val
            elif k in ['ij']:
                low = base_val
                high = base_val * (1 + variation)
            else:
                low = base_val * (1 - variation)
                high = base_val * (1 + variation)
            test_params[k] = np.random.uniform(low, high)
        
        res = run_optimization(test_params)
        if res.F is not None and len(res.F) > 0:
            # Store all objective values from this iteration
            w_values.extend(-res.F[:, 0])  # Revenue (negated back to positive)
            f_values.extend(res.F[:, 1])   # Carbon Emission
            i_values.extend(res.F[:, 2])   # Infrastructure Cost
            s_values.extend(-res.F[:, 3])  # Satisfaction (negated back to positive)
            
            aggregated = [aggregate_obj(ind) for ind in res.F]
            idx_best = np.argmin(aggregated)
            current_best_val = aggregated[idx_best]

            if (best_score is None) or (current_best_val < best_score):
                best_score = current_best_val
                best_params = test_params.copy()
    
    # Print ranges for all objectives
    print("\nObjective Function Ranges Across All Simulations:")
    print("\nRevenue (W):")
    print(f"Min: {min(w_values):,.2f}")
    print(f"Max: {max(w_values):,.2f}")
    print(f"Range: {max(w_values) - min(w_values):,.2f}")
    
    print("\nCarbon Emission (F):")
    print(f"Min: {min(f_values):,.2f}")
    print(f"Max: {max(f_values):,.2f}")
    print(f"Range: {max(f_values) - min(f_values):,.2f}")
    
    print("\nInfrastructure Cost (I):")
    print(f"Min: {min(i_values):,.2f}")
    print(f"Max: {max(i_values):,.2f}")
    print(f"Range: {max(i_values) - min(i_values):,.2f}")
    
    print("\nSatisfaction (S):")
    print(f"Min: {min(s_values):.4f}")
    print(f"Max: {max(s_values):.4f}")
    print(f"Range: {max(s_values) - min(s_values):.4f}")

    return best_params, best_score

# Run base case first
print("\nRunning base case optimization...")
base_results = run_optimization(base_params)

# Run Monte Carlo simulation
print("\nStarting Monte Carlo simulation...")
optimal_params, score = monte_carlo_simulation(base_params, n_samples=50, variation=0.2)

# Print results
print("\nOptimal parameters from Monte Carlo simulation:")
if optimal_params:
    for k, v in optimal_params.items():
        original = base_params[k]
        change = ((v - original) / original) * 100
        print(f"{k}: {v:.4f} (change from base: {change:+.2f}%)")
    print(f"\nAggregate score (lower is better): {score:.4f}")
else:
    print("No feasible solution found.")

# Print optimal solution values
if optimal_params:
    res_opt = run_optimization(optimal_params)
    if res_opt.F is not None and len(res_opt.F) > 0:
        # Find the solution with the best aggregate score
        def aggregate_obj(values):
            coefficients = np.array([0.3372, 0.2091, 0.2882, 0.1655])
            return np.sum(values * coefficients)
        
        aggregated = [aggregate_obj(ind) for ind in res_opt.F]
        best_idx = np.argmin(aggregated)
        
        # Get the best solution
        best_solution = res_opt.X[best_idx]
        best_objectives = res_opt.F[best_idx]
        
        # Calculate the actual W, F, I, S values
        W = -best_objectives[0]  # Revenue (negated back to positive)
        F = best_objectives[1]   # Carbon Emission
        I = best_objectives[2]   # Infrastructure Cost
        S = -best_objectives[3]  # Satisfaction (negated back to positive)
        
        # Get the decision variables v1-v6
        v1, v2, v3, v4, v5, v6 = best_solution
        
        print("\nOptimal Solution Values:")
        print(f"Objectives:")
        print(f"W (Revenue): {W:,.2f}")
        print(f"F (Carbon Emission): {F:,.2f}")
        print(f"I (Infrastructure Cost): {I:,.2f}")
        print(f"S (Satisfaction): {S:.4f}")
        
        print(f"\nDecision Variables:")
        print(f"v1: {v1:,.2f}")
        print(f"v2: {v2:,.2f}")
        print(f"v3: {v3:,.2f}")
        print(f"v4: {v4:,.2f}")
        print(f"v5: {v5:,.2f}")
        print(f"v6: {v6:,.2f}")

# Visualize results
print("\nVisualizing Pareto fronts...")

# Plot base case
if base_results.F is not None and len(base_results.F) > 0:
    plot = Scatter(title="Base Case Pareto Front")
    plot.add(base_results.F[:, :3], s=30, facecolors='lightblue', edgecolors='lightblue', label='Base')
    plot.show()

# Plot optimal case
if optimal_params:
    res_opt = run_optimization(optimal_params)
    if res_opt.F is not None and len(res_opt.F) > 0:
        plot = Scatter(title="Optimal Parameters Pareto Front")
        plot.add(res_opt.F[:, :3], s=30, facecolors='pink', edgecolors='pink', label='Optimal')
        plot.show()