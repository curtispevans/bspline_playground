import numpy as np
from numpy.typing import NDArray

import params as P


class SimpleDynamics:
    def __init__(self, x0: NDArray[np.float64] = np.zeros(3), v0 : float = 0.0):
        """
        Initialize the simple dynamics model.
        Initial state vector [px, py, heading, vx, vy]

        Args:
            x0 (NDArray[np.float64]): Initial state vector.
        """
        self.x = x0
        self.max_g = P.max_g
        self.max_acceleration = P.max_acceleration
        self.Ts = P.Ts
        self.v0 = v0

    def update(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Update the state of the system based on the control input.

        Args:
            u (NDArray[np.float64]): Control input vector.
                [xn]
        Returns:
            x_next (NDArray[np.float64]): State vector at the next time step.
        """
        self.x = self.rk4_step(self.x, u, self.Ts)
        return self.x

    def rk4_step(self, x: NDArray[np.float64], u: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """
        Perform a single rk4 integration step.

        Args:
            x (NDArray[np.float64]): Current state vector.
                [px, py, heading, vx, vy]
            u (NDArray[np.float64]): Control input vector.
                [xn]
            dt (float): Time step for integration.
        Returns:
            x_next (NDArray[np.float64]): State vector at the next time step.
        """
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        x_next = x + (dt / 6) * (k1 + 2 * (k2 + k3) + k4)
        return x_next

    def f(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the state derivatives.

        Args:
            x (1D array): Current state vector [px, py, theta, vx, vy]
            u (1D array): Control input vector [an]
        Returns:
            xdot (NDArray[np.float64]): State derivatives.
        """
        px, py, heading = x
        [a_normal] = np.clip(u, -self.max_acceleration, self.max_acceleration)

        xdot = np.empty_like(x)
        xdot[0] = self.v0 * np.cos(heading)
        xdot[1] = self.v0 * np.sin(heading)
        xdot[2] = a_normal / self.v0
        
        return xdot
 