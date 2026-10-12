import sys
# Tell Python to look in the build directory for the compiled file
sys.path.append('../bsplines_library/build')
import bspline_module
import numpy as np
import matplotlib.pyplot as plt

d = 3
M = 6
spline = bspline_module.UniformBSpline(d, M)
t = np.linspace(0, M, 100)


c_x = np.arange(9)
c_y = np.array([0, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5])
B = np.array([spline.basis_vector(ti) for ti in t]).T

c = np.array([c_x, c_y])
print(c.shape, B[0].shape)
points = c @ B
print(points.shape)
plt.figure(figsize=(12, 6))
plt.plot(points[0], points[1])
plt.plot(c_x, c_y, 'o', color='orange', markersize=4)
plt.title(f"Uniform B-Spline Curve d={d}, M={M}")
plt.savefig(f'uniform_bspline_d{d}_M{M}.png', dpi=300)

plt.show()

