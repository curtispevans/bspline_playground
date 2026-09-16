import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from dynamics import SimpleDynamics
import params as P

x0 = np.array([1000, 0, 0, -100, 0])
vehicle = SimpleDynamics(x0)

t0 = 0
tf = 5
t = 0

# execute a maximum acceleration maneuver
u = np.array([P.max_acceleration])
states = [x0]
R = np.linalg.norm(x0[3:5])**2 / P.max_g
print(f"Max radius of curvature: {R:.2f} m")

states_on_circle = []
while t < tf:
    if t < 1:
        u = np.array([0])
    elif 1 <= t < 2:
        u = np.array([P.max_acceleration])
        states_on_circle.append(states[-1])
    else:
        u = np.array([0])
    x = vehicle.update(u)
    t += P.Ts
    states.append(x)

states = np.array(states)
states_on_circle = np.array(states_on_circle)
plt.plot(states[:, 0], states[:, 1])
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.plot(states_on_circle[:, 0], states_on_circle[:, 1], 'r--', label='Max Acceleration Maneuver')
plt.show()

plt.plot(np.linalg.norm(states[:, 3:5], axis=1))
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed vs Time')
plt.show()


