import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from dynamics import SimpleDynamics
import params as P

def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_control(x_current: NDArray[np.float64], v: float, target: NDArray[np.float64]) -> NDArray[np.float64]:
    a_max = P.max_acceleration
    px, py, heading = x_current
    R_min = (v**2) / a_max

    dx = target[0] - px
    dy = target[1] - py
    dist_to_target = np.hypot(dx, dy)

    # 1. INSIDE THE CIRCLE: Execute max acceleration turn
    if dist_to_target <= R_min:
        # Determine whether left (+a_max) or right (-a_max) turn curves into target
        cross_prod = np.cos(heading) * dy - np.sin(heading) * dx
        turn_sign = 1.0 if cross_prod >= 0 else -1.0
        return np.array([turn_sign * a_max])

    # 2. OUTSIDE THE CIRCLE: Aim for the tangent approach path
    # Angle offset to hit the circle perimeter tangentially
    offset_angle = np.arcsin(np.clip(R_min / dist_to_target, -1.0, 1.0))
    direct_angle = np.arctan2(dy, dx)

    # Pick the tangent line closest to current heading
    target_heading_left = direct_angle + offset_angle
    target_heading_right = direct_angle - offset_angle

    err_left = abs(wrap_angle(target_heading_left - heading))
    err_right = abs(wrap_angle(target_heading_right - heading))

    chosen_heading = target_heading_left if err_left < err_right else target_heading_right
    heading_error = wrap_angle(chosen_heading - heading)

    # Straight line guidance outside the circle
    k_p = 3.0
    an = np.clip(k_p * v * heading_error, -a_max, a_max)
    return np.array([an])

def run_trajectory(vechicle: SimpleDynamics, c: NDArray[np.float64]) -> NDArray[np.float64]:
    t0 = 0
    tf = 60
    t = 0
    states = [vehicle.x]

    while t < tf:
        u = compute_control(vehicle.x, vehicle.v0, c)
        x = vehicle.update(u)
        t += P.Ts
        states.append(x)
        if np.allclose(x[:2], target, atol=10.0):
            print(f"Reached target at time: {t:.2f} s")
            break

    return np.array(states)

x0 = np.array([2000, 0, np.pi])
v0 = 50
vehicle = SimpleDynamics(x0, v0)
R = v0**2 / P.max_acceleration
target = np.array([0, 0])  # Target position at the origin
gamma = 15.0 * np.pi/180.0  # 15 degrees in radians
c = R * np.array([np.cos(gamma), np.sin(gamma)])  # Point on the circle at 15 degrees

t0 = 0
tf = 60
t = 0



# execute a maximum acceleration maneuver
u = np.array([P.max_acceleration])
states = [x0]

print(f"Max radius of curvature: {R:.2f} m")

states_on_circle = []
while t < tf:
    u = compute_control(vehicle.x, vehicle.v0, c)
    x = vehicle.update(u)
    t += P.Ts
    states.append(x)
    if np.allclose(x[:2], target, atol=10.0):
        print(f"Reached target at time: {t:.2f} s")
        break


states = np.array(states)
points = np.linspace(0, 2 * np.pi, 100)
plt.plot(target[0], target[1], 'ro', label='Target')
plt.plot(R * np.cos(points), R * np.sin(points), 'r--', label='Max Turn Radius')
plt.plot(R*np.cos(points) + c[0], R*np.sin(points) + c[1], 'b--', label='Target Circle')
plt.plot(c[0], c[1], 'bo', label='Target Point on Circle')
plt.plot(states[:, 0], states[:, 1])
plt.axis('equal')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.savefig('dubins.png', dpi=300)
plt.show()



