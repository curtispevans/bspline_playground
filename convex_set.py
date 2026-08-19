import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import sys
sys.path.append('../bsplines_library/build')
import bspline_module

# -------------------------------------------------------------
# 1. Analytical System Construction
# -------------------------------------------------------------
def get_analytical_representation(M, d, a, epsilon, start, end):
    n = M + d
    N = n * n
    
    # Equality Constraints: C[:, 0] == start, C[:, -1] == end
    A_eq = np.zeros((2 * n, N))
    b_eq = np.concatenate([start.ravel(), end.ravel()])
    
    for row in range(n):
        A_eq[row, row * n] = 1.0                # First column C[:, 0]
        A_eq[n + row, row * n + (n - 1)] = 1.0  # Last column C[:, -1]
        
    c0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
    V = null_space(A_eq)
    
    # Second-difference inequalities
    num_diff_cols = n - 2
    num_ineqs = n * num_diff_cols
    A_diff = np.zeros((num_ineqs, N))
    
    idx = 0
    for i in range(n):
        for j in range(num_diff_cols):
            A_diff[idx, i * n + j] = 1.0
            A_diff[idx, i * n + (j + 1)] = -2.0
            A_diff[idx, i * n + (j + 2)] = 1.0
            idx += 1
            
    A_ub = np.vstack([A_diff, -A_diff])
    b_ub = np.full(2 * num_ineqs, a - epsilon)
    
    A_tilde = A_ub @ V
    b_tilde = b_ub - A_ub @ c0
    
    return c0, V, A_tilde, b_tilde

# -------------------------------------------------------------
# 2. Sampler: Finds random valid C matrices within the polytope
# -------------------------------------------------------------
def sample_valid_C(c0, V, A_tilde, b_tilde, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    k = V.shape[1]
    alpha = cp.Variable(k)
    
    # Pick a random objective direction to hit diverse boundary/interior points
    random_dir = np.random.randn(k)
    
    constraints = [A_tilde @ alpha <= b_tilde]
    objective = cp.Minimize(random_dir @ alpha + 0.1 * cp.sum_squares(alpha))
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status not in ["optimal", "feasible"]:
        raise ValueError("Could not solve for a valid alpha.")
        
    c_vec = c0 + V @ alpha.value
    return c_vec.reshape((n, n))

# -------------------------------------------------------------
# 3. Trajectory Evaluator (B-Spline Basis Curve)
# -------------------------------------------------------------
def generate_b_spline_trajectory(C, M, d, num_points=200):
    """
    Evaluates C as a set of basis curves over normalized time t in [0, 1].
    Uses Bernstein polynomials (Bezier curves) as the smooth basis.
    """
    t = np.linspace(0, M, num_points)
    clamped = bspline_module.ClampedUniformBSpline(d, M)

    B = np.array([clamped.basis_vector(t_i) for t_i in t]).T
    trajectory = C @ B
    

    return t, trajectory
    

# -------------------------------------------------------------
# 4. Plotting Setup
# -------------------------------------------------------------
M, d = 10, 2
n = M + d
a, epsilon = 5, 0.01

# Define Start state (all zeros) and End state (e.g. linear ramp from 0 to 2)
start_vec = np.ones(n) * 100
start_vec[1] = 0.0  # Start at 0 for the first state
end_vec = np.zeros(n)

c0, V, A_tilde, b_tilde = get_analytical_representation(M, d, a, epsilon, start_vec, end_vec)

# Generate several distinct valid C matrices
num_trajectories = 100
plt.figure(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))

for idx in range(num_trajectories):
    C_sample = sample_valid_C(c0, V, A_tilde, b_tilde, n, seed=idx * 10)
    t, traj = generate_b_spline_trajectory(C_sample, M, d)
    
    # Plot the main state (row 0 of the trajectory matrix)
    plt.plot(traj[0,:], traj[1, :], color=colors[idx], linewidth=2, label=f"Trajectory {idx+1}")
    
    # Plot control points for row 0 as dots
    t_control = np.linspace(0, 1, n)
    plt.scatter(C_sample[0,:], C_sample[1, :], color=colors[idx], s=40, zorder=5)

plt.axhline(start_vec[0], color='black', linestyle='--', alpha=0.5, label='Start Boundary')
plt.axhline(end_vec[0], color='black', linestyle=':', alpha=0.5, label='End Boundary')

plt.title(f"Sampled Valid Trajectories from Feasible Set (M={M}, d={d})", fontsize=12)
plt.xlabel("Normalized Time (t)", fontsize=10)
plt.ylabel("State Value $y_0(t)$", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()