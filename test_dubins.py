import numpy as np
from path_manager import dubins_state_at_time
import matplotlib.pyplot as plt



x0 = np.array([1000, 0, np.pi])
v0 = 100
target = np.array([0, 0, -np.pi/2])  # Target position at the origin

t = np.linspace(0, 100, 300)
paths = []
terminal_headings = np.linspace(-1*np.pi/2, np.pi/2, 50) + np.pi
for heading in terminal_headings:
    target[2] = heading
    path = [dubins_state_at_time(x0, target, ti, v0) for ti in t]
    path = np.array(path)
    paths.append(path)



for path in paths:
    plt.plot(path[:, 0], path[:, 1], 'r-', alpha=0.5)
plt.plot(x0[0], x0[1], 'ro', label='Start')
plt.plot(target[0], target[1], 'go', label='Target')
plt.legend()
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dubins Path")
plt.show()