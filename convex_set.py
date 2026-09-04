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
# def get_analytical_representation(M, d, a, epsilon, start, end):
#     n = M + d
#     N = n * n
    
#     # Equality Constraints: C[:, 0] == start, C[:, -1] == end
#     A_eq = np.zeros((2 * n, N))
#     b_eq = np.concatenate([start.ravel(), end.ravel()])
    
#     for row in range(n):
#         A_eq[row, row * n] = 1.0                # First column C[:, 0]
#         A_eq[n + row, row * n + (n - 1)] = 1.0  # Last column C[:, -1]

#     c0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
#     V = null_space(A_eq)
    
#     # Second-difference inequalities
#     num_diff_cols = n - 2
#     # num_diff_cols = n - 1
#     # num_diff_cols = n - 3
#     num_ineqs = n * num_diff_cols
#     A_diff = np.zeros((num_ineqs, N))
    
#     idx = 0
#     for i in range(n):
#         for j in range(num_diff_cols):
#             A_diff[idx, i * n + j] = 1.0
#             A_diff[idx, i * n + (j + 1)] = -2.0
#             A_diff[idx, i * n + (j + 2)] = 1.0

#             # A_diff[idx, i * n + j] = 1.0
#             # A_diff[idx, i * n + (j + 1)] = -1.0

#             # A_diff[idx, i * n + j] = -1.0
#             # A_diff[idx, i * n + (j + 1)] = 3.0
#             # A_diff[idx, i * n + (j + 2)] = -3.0
#             # A_diff[idx, i * n + (j + 3)] = 1.0
            
#             idx += 1

#     b_diff = np.zeros(num_ineqs)
#     for i in range(n):
#         for j in range(num_diff_cols):
#             # Apply larger bound if evaluating y-coordinate control points
#             b_diff[i * num_diff_cols + j] = (
#                 (a[0] - epsilon) if i < n // 2 else (a[1] - epsilon)
#             )
#     A_ub = np.vstack([A_diff, -A_diff])
#     b_ub = np.concatenate([b_diff, b_diff])
    
#     A_tilde = A_ub @ V
#     b_tilde = b_ub - A_ub @ c0
    
#     return c0, V, A_tilde, b_tilde

# def get_analytical_representation(M, d, a, j_max, epsilon, start, end):
#     n = M + d
#     N = n * n
    
#     # -------------------------------------------------------------
#     # 1. Equality Constraints: C[:, 0] == start, C[:, -1] == end
#     # -------------------------------------------------------------
#     A_eq = np.zeros((2 * n, N))
#     b_eq = np.concatenate([start.ravel(), end.ravel()])
    
#     for row in range(n):
#         A_eq[row, row * n] = 1.0                # First column C[:, 0]
#         A_eq[n + row, row * n + (n - 1)] = 1.0  # Last column C[:, -1]

#     c0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
#     V = null_space(A_eq)
    
#     # -------------------------------------------------------------
#     # 2. Second-Difference Inequalities (Acceleration Limits)
#     # -------------------------------------------------------------
#     num_diff2_cols = n - 2
#     num_ineqs2 = n * num_diff2_cols
#     A_diff2 = np.zeros((num_ineqs2, N))
    
#     idx = 0
#     for i in range(n):
#         for j in range(num_diff2_cols):
#             A_diff2[idx, i * n + j] = 1.0
#             A_diff2[idx, i * n + (j + 1)] = -2.0
#             A_diff2[idx, i * n + (j + 2)] = 1.0
#             idx += 1

#     b_diff2 = np.zeros(num_ineqs2)
#     for i in range(n):
#         for j in range(num_diff2_cols):
#             b_diff2[i * num_diff2_cols + j] = (
#                 (a[0] - epsilon) if i < n // 2 else (a[1] - epsilon)
#             )

#     # -------------------------------------------------------------
#     # 3. Third-Difference Inequalities (Jerk Limits)
#     # -------------------------------------------------------------
#     num_diff3_cols = n - 3
#     num_ineqs3 = n * num_diff3_cols
#     A_diff3 = np.zeros((num_ineqs3, N))
    
#     idx = 0
#     for i in range(n):
#         for j in range(num_diff3_cols):
#             A_diff3[idx, i * n + j] = -1.0
#             A_diff3[idx, i * n + (j + 1)] = 3.0
#             A_diff3[idx, i * n + (j + 2)] = -3.0
#             A_diff3[idx, i * n + (j + 3)] = 1.0
#             idx += 1

#     b_diff3 = np.zeros(num_ineqs3)
#     for i in range(n):
#         for j in range(num_diff3_cols):
#             b_diff3[i * num_diff3_cols + j] = (
#                 (j_max[0] - epsilon) if i < n // 2 else (j_max[1] - epsilon)
#             )

#     # -------------------------------------------------------------
#     # 4. Stack Acceleration and Jerk Constraints
#     # -------------------------------------------------------------
#     # Upper/lower bounds for acceleration and jerk combined
#     A_diff = np.vstack([A_diff2, A_diff3])
#     b_diff = np.concatenate([b_diff2, b_diff3])

#     A_ub = np.vstack([A_diff, -A_diff])
#     b_ub = np.concatenate([b_diff, b_diff])
    
#     # Transform to reduced basis coordinates (alpha space)
#     A_tilde = A_ub @ V
#     b_tilde = b_ub - A_ub @ c0
    
#     return c0, V, A_tilde, b_tilde

def get_analytical_representation_2D(M, d, a, j_max, epsilon, start, end):
    n = M + d
    N = 2 * n  # c_vec[:n] is X, c_vec[n:] is Y
    
    # -------------------------------------------------------------
    # 1. Equality Constraints: C[0, 0]=x_start, C[0, -1]=x_end, etc.
    # -------------------------------------------------------------
    A_eq = np.zeros((4, N))
    b_eq = np.array([start[0], end[0], start[1], end[1]])
    
    # X start & end (indices 0 and n-1)
    A_eq[0, 0] = 1.0          
    A_eq[1, n - 1] = 1.0      
    
    # Y start & end (indices n and 2n-1)
    A_eq[2, n] = 1.0          
    A_eq[3, 2 * n - 1] = 1.0  

    c0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
    V = null_space(A_eq)
    
    # -------------------------------------------------------------
    # 2. Acceleration Constraints (Second Differences)
    # -------------------------------------------------------------
    num_diff2 = n - 2
    A_diff2 = np.zeros((2 * num_diff2, N))
    b_diff2 = np.zeros(2 * num_diff2)
    
    # X acceleration (limit = a[0])
    for j in range(num_diff2):
        A_diff2[j, j:j+3] = [1.0, -2.0, 1.0]
        b_diff2[j] = a[0] - epsilon

    # Y acceleration (limit = a[1])
    for j in range(num_diff2):
        idx = num_diff2 + j
        A_diff2[idx, n + j : n + j + 3] = [1.0, -2.0, 1.0]
        b_diff2[idx] = a[1] - epsilon

    # -------------------------------------------------------------
    # 3. Jerk Constraints (Third Differences)
    # -------------------------------------------------------------
    num_diff3 = n - 3
    A_diff3 = np.zeros((2 * num_diff3, N))
    b_diff3 = np.zeros(2 * num_diff3)
    
    # X jerk (limit = j_max[0])
    for j in range(num_diff3):
        A_diff3[j, j:j+4] = [-1.0, 3.0, -3.0, 1.0]
        b_diff3[j] = j_max[0] - epsilon

    # Y jerk (limit = j_max[1])
    for j in range(num_diff3):
        idx = num_diff3 + j
        A_diff3[idx, n + j : n + j + 4] = [-1.0, 3.0, -3.0, 1.0]
        b_diff3[idx] = j_max[1] - epsilon

    # -------------------------------------------------------------
    # 4. Stack Constraints
    # -------------------------------------------------------------
    A_diff = np.vstack([A_diff2, A_diff3])
    b_diff = np.concatenate([b_diff2, b_diff3])

    A_ub = np.vstack([A_diff, -A_diff])
    b_ub = np.concatenate([b_diff, b_diff])
    
    A_tilde = A_ub @ V
    b_tilde = b_ub - A_ub @ c0
    
    return c0, V, A_tilde, b_tilde

def get_analytical_representation_2D_no_jerk(M, d, a, num_epsilon, epsilon, start, end):
    n = M + d
    N = 2 * n  # c_vec[:n] is X, c_vec[n:] is Y
    
    # -------------------------------------------------------------
    # 1. Equality Constraints: C[0, 0]=x_start, C[0, -1]=x_end, etc.
    # -------------------------------------------------------------
    A_eq = np.zeros((4, N))
    b_eq = np.array([start[0], end[0], start[1], end[1]])
    
    # X start & end (indices 0 and n-1)
    A_eq[0, 0] = 1.0          
    A_eq[1, n - 1] = 1.0      
    
    # Y start & end (indices n and 2n-1)
    A_eq[2, n] = 1.0          
    A_eq[3, 2 * n - 1] = 1.0  

    c0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
    V = null_space(A_eq)
    
    # -------------------------------------------------------------
    # 2. Acceleration Constraints (Second Differences)
    # -------------------------------------------------------------
    num_diff2 = n - 2
    A_diff2 = np.zeros((2 * num_diff2, N))
    b_diff2 = np.zeros(2 * num_diff2)
    
    # X acceleration (limit = a[0])
    for j in range(num_diff2):
        A_diff2[j, j:j+3] = [1.0, -2.0, 1.0]
        if j < num_epsilon:  # Apply tighter limit for the first few control points
            b_diff2[j] = epsilon*0
        else:
            b_diff2[j] = a[0] - epsilon

    # Y acceleration (limit = a[1])
    for j in range(num_diff2):
        idx = num_diff2 + j
        A_diff2[idx, n + j : n + j + 3] = [1.0, -2.0, 1.0]
        if j < num_epsilon:  # Apply tighter limit for the first few control points
            b_diff2[idx] = epsilon*0
        else:
            b_diff2[idx] = a[1] - epsilon

    # -------------------------------------------------------------
    # 4. Stack Constraints
    # -------------------------------------------------------------

    A_ub = np.vstack([A_diff2, -A_diff2])
    b_ub = np.concatenate([b_diff2, b_diff2])
    
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
    # Slice explicitly: Row 0 is X, Row 1 is Y
    X_pts = c_vec[:n]
    Y_pts = c_vec[n:]
    
    return np.vstack([X_pts, Y_pts])
    

def sample_valid_C_with_target_y(
    c0, V, A_tilde, b_tilde, n, target_y_peak, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    k = V.shape[1]
    alpha = cp.Variable(k)

    # Reconstruct C from alpha inside CVXPY
    c_vec = c0 + V @ alpha
    C = cp.reshape(c_vec, (n, n), order="C")

    # Force or encourage mid-trajectory y control points toward target_y_peak
    mid_idx = n // 2
    objective = cp.Minimize(cp.square(C[1, mid_idx] - target_y_peak))

    constraints = [A_tilde @ alpha <= b_tilde]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
    
    if prob.status not in ["optimal", "feasible"]:
        raise ValueError("Optimization failed to find a valid solution.")
        
    return C.value

def sample_diverse_C(c0, V, A_tilde, b_tilde, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    k = V.shape[1]
    alpha = cp.Variable(k)
    c_vec = c0 + V @ alpha
    C = cp.reshape(c_vec, (n, n), order='C')
    
    # 1. Pick a random weight specifically targeting Y-axis control points
    y_weights = np.random.uniform(-1.0, 1.0, size=n)
    
    # 2. Objective maximizes/minimizes a weighted combination of Y control points
    objective = cp.Minimize(y_weights @ C[1, :] + 1e-3 * cp.sum_squares(alpha))
    
    constraints = [A_tilde @ alpha <= b_tilde]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
    
    if prob.status not in ["optimal", "feasible"]:
        raise ValueError("Optimization failed to find a valid solution.")
        
    return C.value

def sample_diverse_C_2D(c0, V, A_tilde, b_tilde, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    k = V.shape[1]
    alpha = cp.Variable(k)
    c_vec = c0 + V @ alpha
    
    # Explicitly separate X (first n) and Y (last n) to prevent order bugs
    X_pts = c_vec[:n]
    Y_pts = c_vec[n:]
    C = cp.vstack([X_pts, Y_pts])
    
    y_weights = np.random.uniform(-1.0, 1.0, size=n)
    objective = cp.Minimize(y_weights @ Y_pts + 1e-3 * cp.sum_squares(alpha))
    
    constraints = [A_tilde @ alpha <= b_tilde]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
    
    if prob.status not in ["optimal", "feasible"]:
        raise ValueError("Optimization failed.")
        
    return C.value

def sample_hit_and_run(c0, V, A_tilde, b_tilde, n, num_samples=30, burn_in=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    k = V.shape[1]
    
    # 1. Find an initial strictly interior point using Chebyshev center
    alpha_var = cp.Variable(k)
    r = cp.Variable()
    
    # Normalize rows of A_tilde for accurate ball radius r
    norms = np.linalg.norm(A_tilde, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    A_norm = A_tilde / norms
    
    prob = cp.Problem(cp.Maximize(r), [A_norm @ alpha_var + r <= b_tilde / norms.ravel()])
    prob.solve()
    
    current_alpha = alpha_var.value.copy()
    
    # 2. Burn-in steps to lose initial bias
    for _ in range(burn_in):
        direction = np.random.randn(k)
        direction /= np.linalg.norm(direction)
        
        # Calculate line-search step limits: A_tilde (alpha + t * dir) <= b_tilde
        Ad = A_tilde @ direction
        slack = b_tilde - A_tilde @ current_alpha
        
        pos_mask = Ad > 1e-9
        neg_mask = Ad < -1e-9
        
        t_max = np.min(slack[pos_mask] / Ad[pos_mask]) if np.any(pos_mask) else 100.0
        t_min = np.max(slack[neg_mask] / Ad[neg_mask]) if np.any(neg_mask) else -100.0
        
        if t_max > t_min:
            t = np.random.uniform(t_min, t_max)
            current_alpha += t * direction

    # 3. Collect diverse interior samples
    trajectories = []
    for _ in range(num_samples):
        direction = np.random.randn(k)
        direction /= np.linalg.norm(direction)
        
        Ad = A_tilde @ direction
        slack = b_tilde - A_tilde @ current_alpha
        
        pos_mask = Ad > 1e-9
        neg_mask = Ad < -1e-9
        
        t_max = np.min(slack[pos_mask] / Ad[pos_mask]) if np.any(pos_mask) else 100.0
        t_min = np.max(slack[neg_mask] / Ad[neg_mask]) if np.any(neg_mask) else -100.0
        
        if t_max > t_min:
            t = np.random.uniform(t_min, t_max)
            current_alpha += t * direction
            
        c_vec = c0 + V @ current_alpha
        trajectories.append(np.vstack([c_vec[:n], c_vec[n:]]))
        
    return trajectories

def sample_convex_combination(c0, V, A_tilde, b_tilde, n, num_vertices=6, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    k = V.shape[1]
    vertices = []
    
    # 1. Generate diverse extreme boundary points
    for i in range(num_vertices):
        alpha = cp.Variable(k)
        
        # Target random weightings on Y-control points to hit outer boundaries
        y_weights = np.random.uniform(-1.0, 1.0, size=n)
        
        # Extract Y entries from alpha representation
        # c_vec[:n] is X, c_vec[n:] is Y
        c_vec = c0 + V @ alpha
        Y_pts = c_vec[n:]
        
        objective = cp.Minimize(y_weights @ Y_pts + 1e-3 * cp.sum_squares(alpha))
        prob = cp.Problem(objective, [A_tilde @ alpha <= b_tilde])
        prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
        
        if prob.status in ["optimal", "feasible"]:
            vertices.append(alpha.value)
            
    if len(vertices) == 0:
        raise ValueError("Could not find valid boundary vertices.")

    # 2. Blend vertices using Dirichlet distribution (guarantees sum = 1, w_i >= 0)
    # weights = np.random.dirichlet(np.ones(len(vertices)))
    weights = np.random.dirichlet(np.full(len(vertices), 0.3))
    # weights = np.random.dirichlet(np.full(len(vertices), 5))
    interior_alpha = sum(w * v for w, v in zip(weights, vertices))
    
    # 3. Construct 2 x n Control Point Matrix
    c_vec_final = c0 + V @ interior_alpha
    X_pts = c_vec_final[:n]
    Y_pts = c_vec_final[n:]
    
    return np.vstack([X_pts, Y_pts])
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
M, d = 8, 3
n = M + d
a = np.array([0.2, 50.])
j_max = np.array([1, 6])
epsilon = 0.01
num_epsilon = 5  # Number of initial control points with tighter constraints


# start_vec = np.ones(n) * 1000
# start_vec[1:] = 0.0  # Start at 0 for the first state
# end_vec = np.zeros(n)

start_vec = np.array([1000.0, 0.0])
end_vec = np.array([0.0, 0.0])


# c0, V, A_tilde, b_tilde = get_analytical_representation_2D(M, d, a, j_max, epsilon, start_vec, end_vec)
c0, V, A_tilde, b_tilde = get_analytical_representation_2D_no_jerk(M, d, a, num_epsilon, epsilon, start_vec, end_vec)

# Generate several distinct valid C matrices
num_trajectories = 100
plt.figure(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))

# control_matrices = sample_hit_and_run(c0, V, A_tilde, b_tilde, n, num_samples=num_trajectories, burn_in=200, seed=42)

for idx in range(num_trajectories):
    # C_sample = sample_valid_C(c0, V, A_tilde, b_tilde, n, seed=idx * 10)
    # C_sample = sample_diverse_C(c0, V, A_tilde, b_tilde, n, seed=idx * 10)
    C_sample = sample_convex_combination(c0, V, A_tilde, b_tilde, n, num_vertices=6, seed=idx * 10)
    # C_sample = sample_diverse_C_2D(c0, V, A_tilde, b_tilde, n, seed=idx * 10)
    # C_sample = sample_hit_and_run(c0, V, A_tilde, b_tilde, n, num_samples=1, burn_in=200, seed=idx * 10)[0]
    # C_sample = control_matrices[idx]
    t, traj = generate_b_spline_trajectory(C_sample, M, d)
    
    # Plot the main state (row 0 of the trajectory matrix)
    plt.plot(traj[0,:], traj[1, :], color=colors[idx], linewidth=2, label=f"Trajectory {idx+1}", alpha=0.7)
    
    # Plot control points
    plt.scatter(C_sample[0,:], C_sample[1, :], color=colors[idx], s=40, zorder=5, alpha=0.5)


plt.title(f"Sampled Valid Trajectories from Feasible Set (M={M}, d={d})", fontsize=12)
plt.xlabel("$x(t)$", fontsize=10)
plt.ylabel("$y(t)$", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend()
plt.tight_layout()
plt.savefig("sampled_trajectories.png", dpi=300)
plt.show()