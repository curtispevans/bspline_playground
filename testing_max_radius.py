import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from dynamics import SimpleDynamics
import params as P
from path_manager import PathManager

def compute_trajectory(x0: NDArray[np.float64], target: NDArray[np.float64], tf: float, v0: float) -> NDArray[np.float64]:
    """
    Compute the trajectory of the vehicle towards the target.

    Args:
        vehicle (SimpleDynamics): The vehicle dynamics model.
        path_manager (PathManager): The path manager for Dubins path.
        target (NDArray[np.float64]): Target position [px, py, heading].
        tf (float): Final time for simulation.

    Returns:
        NDArray[np.float64]: Array of states over time.
    """
    vehicle = SimpleDynamics(x0, v0=v0)
    R = v0**2 / P.max_acceleration
    path_manager = PathManager(R=R, p_s=x0[:2], chi_s=x0[2], p_e=target[:2], chi_e=target[2])
    t = 0
    states = [vehicle.x]

    while t < tf and not path_manager.done:
        u = path_manager.update(vehicle)
        x = vehicle.update(u)
        t += P.Ts
        states.append(x)

    print(f"Reached target at time: {t:.2f} s")
    return np.array(states)


x0 = np.array([3500, 0, np.pi])
v0 = 50
target = np.array([0, 0, -3*np.pi/4])  # Target position at the origin

t0 = 0
tf = 150
t = 0

# execute a maximum acceleration maneuver
states = []
final_headings = np.linspace(-np.pi, np.pi, 100)  # Test different final headings]
for headings in final_headings:
    target[2] = headings
    states.append((compute_trajectory(x0, target, tf, v0), headings))

R = v0**2 / P.max_acceleration
print(f"Max radius of curvature: {R:.2f} m")



for state in states[::-1]:
    state, heading = state
    plt.plot(state[:, 0], state[:, 1], 'r-', label=f"Heading: {heading} rad", alpha=0.5)
# points = np.linspace(0, 2 * np.pi, 100)
plt.plot(target[0], target[1], 'ro', label='Target')
# plt.plot(path_manager.c_s[0] + R * np.cos(points), path_manager.c_s[1] + R * np.sin(points), "r--", label="Start circle",)
# plt.plot(path_manager.c_e[0] + R * np.cos(points), path_manager.c_e[1] + R * np.sin(points), "g--", label="End circle",)
# plt.plot([path_manager.z1[0], path_manager.z2[0]], [path_manager.z1[1], path_manager.z2[1]], "k--", label="Tangent line",)
# plt.scatter([path_manager.z1[0], path_manager.z2[0]], [path_manager.z1[1], path_manager.z2[1]], label="Tangent points",)
# plt.plot(states[:, 0], states[:, 1])
plt.axis('equal')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
# plt.legend()
plt.savefig('dubins.png', dpi=300)
plt.show()



