import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../bsplines_library/build')
import bspline_module

clamped = bspline_module.ClampedUniformBSpline(2, 10)

print(clamped.knots)
print(clamped.basis_vector(0.0))


