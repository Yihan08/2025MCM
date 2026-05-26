import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Model coefficients
a_f, b_f, c_f, d_f = 0.295, -3.637e-7, 1.243e-13, 6.505e-7
a_i, b_i, c_i = 53.5612, 1.33e13, 2.756e8
a_s, b_s, c_s, d_s = -12.337, -3.921e-12, 155.198, 1.992e-5

# Prior weights
w1, w2, w3, w4 = 0.30918855, 0.17976395, 0.31932943, 0.19171808

def calculate_E(v):
    """Calculate additional income based on visitor numbers"""
    return 20.1741 * v + 17883749.1410

class MultiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=1,  
            n_obj=4,  
            n_ieq_constr=4,
            xl=np.array([1120324]),
            xu=np.array([1650149])
        )

    def _evaluate(self, V, out, *args, **kwargs):
        v = V[:, 0]
        
        # Calculate E for each v
        E = np.array([calculate_E(vi) for vi in v])

        # Calculate objectives with modified F and I
        W = 52 * v + 76502404
        F = a_f * v + b_f * v**2 + c_f * v**3 + d_f - np.minimum(w2 * (E/42), 5000)  # Modified Carbon Emission
        I = a_i * v + b_i * (1 / v) + c_i - w3 * E  # Modified Infrastructure Cost
        S = a_s * np.log(1 + v) + b_s * v**2 + c_s + d_s * v + w4 * E / 280000000 # Modified Satisfaction

        out["F"] = np.column_stack([-W, F, I, -S])

        g = np.column_stack([
            F - 60000,
            I - 450000000,
            0.738 - S,
            300000000 - I,
            #S - 1
        ])
        out["G"] = g

def critic_weight_calculation(X):
    """Calculate weights using CRITIC method"""
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    std_devs = np.std(X_norm, axis=0)
    correlation_matrix = np.corrcoef(X_norm, rowvar=False)
    conflict_intensity = 1 - correlation_matrix
    C = std_devs * np.sum(conflict_intensity, axis=1)
    weights = C / np.sum(C)
    return weights

def topsis_analysis(normalized_objectives, weights):
    """Perform TOPSIS analysis"""
    weighted_matrix = normalized_objectives * weights
    print("A:", weighted_matrix)
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)
    
    distance_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution)**2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution)**2, axis=1))
    print("B:", distance_ideal)
    
    relative_closeness = distance_negative / (distance_ideal + distance_negative)
    best_index = np.argmax(relative_closeness)
    
    return best_index, relative_closeness

# Run optimization
algorithm = NSGA2(pop_size=100)
problem = MultiObjectiveProblem()
res = minimize(problem, algorithm, ("n_gen", 200), seed=42, verbose=True)

if res.F is not None and len(res.F) > 0:
    # Get objectives
    objectives = np.column_stack([
        -res.F[:, 0],
        res.F[:, 1],
        res.F[:, 2],
        -res.F[:, 3]
    ])
    
    # Calculate new CRITIC weights
    new_weights = critic_weight_calculation(objectives)
    
    print("\n=== New CRITIC Weights ===")
    print(f"w1 (Revenue): {new_weights[0]:.4f}")
    print(f"w2 (Carbon Emission): {new_weights[1]:.4f}")
    print(f"w3 (Infrastructure): {new_weights[2]:.4f}")
    print(f"w4 (Satisfaction): {new_weights[3]:.4f}")
    
    # TOPSIS analysis with new weights
    scaler = MinMaxScaler()
    normalized_objectives = scaler.fit_transform(objectives)
    best_index, topsis_scores = topsis_analysis(normalized_objectives, new_weights)
    
    # Print best solution
    print("\n=== Best Solution According to TOPSIS Analysis ===")
    optimal_V = res.X[best_index][0]
    optimal_E = calculate_E(optimal_V)
    
    optimal_W = 52 * optimal_V + 76502404 + w1 * optimal_E
    optimal_F = a_f * optimal_V + b_f * optimal_V**2 + c_f * optimal_V**3 + d_f - np.minimum(w2 * (optimal_E/42), 5000)
    optimal_I = a_i * optimal_V + b_i * (1 / optimal_V) + c_i - w3 * optimal_E
    optimal_S = a_s * np.log(1 + optimal_V) + b_s * optimal_V**2 + c_s + d_s * optimal_V + w4 * optimal_E / 280000000
    
    print(f"Number of Visitors (V): {optimal_V:.0f}")
    print(f"Revenue (W): {optimal_W:.2f}")
    print(f"Carbon Emission (F): {optimal_F:.2f}")
    print(f"Infrastructure Cost (I): {optimal_I:.2f}")
    print(f"Satisfaction (S): {optimal_S:.4f}")
    print(f"TOPSIS Score: {topsis_scores[best_index]:.4f}")
    
    # 3D Pareto Front Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the first three objectives (W, F, I)
    objectives_3d = res.F[:, :3]
    
    # Plot Pareto solutions first
    ax.scatter(objectives_3d[:, 0], 
              objectives_3d[:, 1], 
              objectives_3d[:, 2],
              color='lightblue',
              s=60,
              alpha=0.6,
              label='Pareto Solutions')
    
    # Then plot the best solution to ensure it's not covered
    ax.scatter(objectives_3d[best_index, 0],
              objectives_3d[best_index, 1],
              objectives_3d[best_index, 2],
              color='black',
              s=60,  # Slightly larger size for visibility
              label='Best Solution (TOPSIS)',
              zorder=2)  # Ensure it's plotted on top
    
    # Labels and title
    ax.set_xlabel('Revenue (-W)')
    ax.set_ylabel('Carbon Emission (F)')
    ax.set_zlabel('Infrastructure Cost (I)')
    ax.set_title('Pareto Front with Extra Revenue', pad=20)
    
    # Add legend
    ax.legend()
    
    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

else:
    print("No feasible solutions found.")