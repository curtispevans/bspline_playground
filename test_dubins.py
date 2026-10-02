import numpy as np
from path_manager import dubins_state_at_time
import matplotlib.pyplot as plt



x0 = np.array([3000, 0, np.pi])
v0 = 50
target = np.array([0, 0, -3*np.pi/4])  # Target position at the origin

t = np.linspace(0, 100, 200)
path = [dubins_state_at_time(x0, target, ti, v0)[0] for ti in t]
path = np.array(path)

plt.plot(path[:, 0], path[:, 1])
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dubins Path")
plt.show()