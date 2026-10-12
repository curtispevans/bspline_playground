"""Proportional-navigation guidance for ``SimpleDynamics``.

``SimpleDynamics`` takes a one-element control ``[a_normal]`` and applies
heading_rate = a_normal / speed. ``pronav`` returns a control in that format.
"""

from math import atan2, cos, hypot, pi, sin
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


def pronav(
    robot_state: Sequence[float],
    ghost_state: Sequence[float],
    robot_speed: float,
    ghost_speed: float = 0.0,
    navigation_constant: float = 3.0,
    max_acceleration: Optional[float] = None,
) -> NDArray[np.float64]:
    """Return the normal-acceleration control that steers toward a ghost.

    Args:
        robot_state: Robot pose ``(px, py, heading)`` as used by
            ``SimpleDynamics.x``. Heading is radians CCW from +x.
        ghost_state: Ghost pose ``(px, py, heading)``. The ghost heading is
            used with ``ghost_speed`` to estimate its velocity. For a fixed
            target, leave ``ghost_speed=0``.
        robot_speed: The robot's constant speed (``SimpleDynamics.v0``).
        ghost_speed: Ghost speed along its heading. Supply the speed of the
            predicted ghost trajectory if it is moving.
        navigation_constant: Dimensionless proportional-navigation gain.
        max_acceleration: Optional symmetric acceleration limit. Pass
            ``dynamics.max_acceleration`` (or ``P.max_acceleration``) to
            match the dynamics model's saturation.

    Returns:
        A one-element NumPy array ``[a_normal]`` suitable for
        ``dynamics.update(u)``. Positive acceleration turns left.

    The controller computes ``a_normal = N * closing_speed * LOS_rate``.
    If the target is not closing yet, it uses pure pursuit to turn toward its
    current position, which lets it recover when the target begins behind it.
    Call this at every control step with the current robot pose and current
    ghost pose.
    """
    if len(robot_state) != 3:
        raise ValueError("robot_state must be (px, py, heading)")
    if len(ghost_state) != 3:
        raise ValueError("ghost_state must be (px, py, heading)")
    if robot_speed <= 0.0:
        raise ValueError("robot_speed must be positive")
    if ghost_speed < 0.0:
        raise ValueError("ghost_speed must be nonnegative")
    if navigation_constant < 0.0:
        raise ValueError("navigation_constant must be nonnegative")
    if max_acceleration is not None and max_acceleration < 0.0:
        raise ValueError("max_acceleration must be nonnegative")

    px, py, heading = robot_state
    gx, gy, ghost_heading = ghost_state

    rx, ry = gx - px, gy - py
    distance = hypot(rx, ry)
    if distance <= 1e-12:
        return np.array([0.0], dtype=float)

    # Relative velocity = ghost velocity - robot velocity.
    rel_vx = ghost_speed * cos(ghost_heading) - robot_speed * cos(heading)
    rel_vy = ghost_speed * sin(ghost_heading) - robot_speed * sin(heading)

    # LOS angular rate and range closing speed.
    los_rate = (rx * rel_vy - ry * rel_vx) / (distance * distance)
    closing_speed = -(rx * rel_vx + ry * rel_vy) / distance

    if closing_speed <= 0.0:
        # Pure-pursuit fallback: omega = v * curvature, then a_n = v * omega.
        bearing_error = atan2(ry, rx) - heading
        bearing_error = (bearing_error + pi) % (2.0 * pi) - pi
        if abs(abs(bearing_error) - pi) <= 1e-12:
            # At exactly 180 degrees pure pursuit is otherwise zero; choose
            # a deterministic turn direction to get the robot moving.
            normal_acceleration = -2.0 * robot_speed * robot_speed / distance
        else:
            normal_acceleration = (
                2.0 * robot_speed * robot_speed * sin(bearing_error) / distance
            )
    else:
        normal_acceleration = (
            navigation_constant * closing_speed * los_rate
        )

    if max_acceleration is not None:
        normal_acceleration = max(
            -max_acceleration, min(max_acceleration, normal_acceleration)
        )
    return np.array([normal_acceleration], dtype=float)
