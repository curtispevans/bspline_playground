import sys
import numpy as np
import matplotlib.pyplot as plt
from speed_profile import evolve_trajectory
sys.path.append('../bsplines_library/build')
import bspline_module


d = 3
M = 10
clamped = bspline_module.ClampedUniformBSpline(d, M)
clamped_d_1 = bspline_module.ClampedUniformBSpline(d-1, M)

print(clamped.knots)
print(f'bd shape {len(clamped.basis_vector(0.0))}')
print(f'bd_1 shape {len(clamped_d_1.basis_vector(0.0))}')


first_last = [d/i for i in range(1,d)]
middle = [1]*(M-d+1)
diag_elements = first_last + middle + first_last[::-1]
D_bar = np.diag(diag_elements)

zero_row = np.zeros((1, M+d-1))

print(f'D_bar shape {D_bar.shape}')
D = -np.block([[D_bar], [zero_row]]) + np.block([[zero_row], [D_bar]])
print(f'D shape {D.shape}')

c_points_x = np.arange(M+d)
c_points_y = c_points_x**2

C = np.vstack([c_points_x, c_points_y])

print(C)
speed_profile = lambda t: 10.0  # Example speed profile
t_grid, tau = evolve_trajectory(C, D, clamped_d_1.basis_vector, speed_profile, M)
print(tau[0], tau[-1], t_grid[0], t_grid[-1])

# t = np.linspace(0, M, 200)
# normal_time_bspline = np.column_stack([C @ np.array(clamped.basis_vector(t_i)) for t_i in t])
# speed_profile_bspline = np.column_stack([C @ np.array(clamped.basis_vector(tau_i)) for tau_i in tau])
# plt.figure(1)
# plt.plot(normal_time_bspline[0], normal_time_bspline[1])
# plt.plot(speed_profile_bspline[0], speed_profile_bspline[1])
# plt.plot(C[0], C[1], 'ro')


# plt.figure(2)
# plt.plot(t_grid, tau, label=r'$\tau(t)$ (Speed-Profile Time)')
# plt.plot(t_grid, np.linspace(0, M, len(t_grid)), '--', label=r'$\tau = t$ (Uniform Time)')
# plt.xlabel('Time $t$ (seconds)')
# plt.ylabel(r'Path Parameter $\tau$')
# plt.title(r'Time Evolution $\tau(t)$ under Speed Profile')
# plt.legend()
# plt.grid(True)
# plt.show()

# 1. Evaluate spatial coordinates along tau(t)
# t_grid and tau come from evolve_trajectory(...)
p_speed = np.column_stack([C @ clamped.basis_vector(tau_i) for tau_i in tau])
x_speed, y_speed = p_speed[0], p_speed[1]

# 2. Evaluate a baseline trajectory with linear time progression
tau_linear = np.linspace(0, M, len(t_grid))
p_linear = np.column_stack([C @ clamped.basis_vector(tau_i) for tau_i in tau_linear])
x_linear, y_linear = p_linear[0], p_linear[1]

# 3. Plot in 3D (X, Y, Time)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the speed-profile trajectory
ax.plot(x_speed, y_speed, t_grid, 'b-', linewidth=2.5, label='Dynamic Speed Profile $v_d(t)$')

# Plot the linear baseline trajectory for comparison
ax.plot(x_linear, y_linear, t_grid, 'r--', linewidth=1.5, label='Linear Progress ($\tau \propto t$)')

# Plot 2D spatial footprint projection on the bottom plane (t = 0)
ax.plot(x_speed, y_speed, np.zeros_like(t_grid), 'k:', alpha=0.5, label='2D Spatial Path')

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Time $t$ (s)')
ax.set_title('3D Space-Time Trajectory $(x, y, t)$')
ax.legend()
plt.show()
