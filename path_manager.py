import numpy as np
from numpy.typing import NDArray
import params as P
from path_follower import get_dubins_parameters, rotation_matrix
from dynamics import SimpleDynamics
from pronav import pronav


class MaxAccelerationPronavPathManager:
    """Apply maximum normal acceleration, then PN toward a fixed goal.

    Call ``update(vehicle)`` once before each ``vehicle.update(control)``.
    The maneuver lasts ``maneuver_duration`` seconds, rounded up to the next
    control step. ``turn_direction=1`` turns left (CCW); ``-1`` turns right.
    Speed stays constant. PN targets only the goal position, with no prescribed
    arrival heading, and uses the vehicle's normal-acceleration limit.

    ``path_state`` is ``"maneuver"``, ``"pronav"``, or ``"complete"``.
    ``switch_time`` records the actual elapsed time of the first PN command.
    Reaching ``goal_radius`` completes either phase. A completed manager returns
    zero normal acceleration; the caller should stop advancing the vehicle.
    """

    def __init__(
        self,
        goal_position: NDArray[np.float64],
        maneuver_duration: float = 2.0,
        turn_direction: int = 1,
        navigation_constant: float = 3.0,
        goal_radius: float = 10.0,
    ):
        goal_position = np.asarray(goal_position, dtype=float)
        if goal_position.shape != (2,) or not np.all(np.isfinite(goal_position)):
            raise ValueError("goal_position must contain two finite coordinates")
        if not np.isfinite(maneuver_duration) or maneuver_duration < 0:
            raise ValueError("maneuver_duration must be finite and nonnegative")
        if turn_direction not in (-1, 1):
            raise ValueError("turn_direction must be +1 (left) or -1 (right)")
        if not np.isfinite(navigation_constant) or navigation_constant <= 0:
            raise ValueError("navigation_constant must be finite and positive")
        if not np.isfinite(goal_radius) or goal_radius <= 0:
            raise ValueError("goal_radius must be finite and positive")

        self.goal_state = np.r_[goal_position, 0.0]
        self.maneuver_duration = float(maneuver_duration)
        self.turn_direction = int(turn_direction)
        self.navigation_constant = float(navigation_constant)
        self.goal_radius = float(goal_radius)
        self.elapsed_time = 0.0
        self.switch_time = None
        self.path_state = "maneuver" if maneuver_duration > 0 else "pronav"
        self.done = False

    def update(self, vehicle: SimpleDynamics) -> NDArray[np.float64]:
        if self.done:
            return np.array([0.0])
        if (not np.all(np.isfinite(
                [vehicle.v0, vehicle.max_acceleration, vehicle.Ts]))
                or min(vehicle.v0, vehicle.max_acceleration, vehicle.Ts) <= 0):
            raise ValueError(
                "Vehicle speed, acceleration limit and step must be positive and finite"
            )

        if np.linalg.norm(vehicle.x[:2] - self.goal_state[:2]) < self.goal_radius:
            self.path_state = "complete"
            self.done = True
            return np.array([0.0])

        # Allow for roundoff from accumulating integration steps at a deadline.
        if (self.path_state == "maneuver" and
                self.elapsed_time + 1e-12 >= self.maneuver_duration):
            self.path_state = "pronav"

        if self.path_state == "maneuver":
            control = np.array([self.turn_direction * vehicle.max_acceleration])
        else:
            if self.switch_time is None:
                self.switch_time = self.elapsed_time
            control = pronav(
                vehicle.x, self.goal_state, robot_speed=vehicle.v0,
                ghost_speed=0.0, navigation_constant=self.navigation_constant,
                max_acceleration=vehicle.max_acceleration,
            )

        self.elapsed_time += vehicle.Ts
        return control


class PathManager:
    def __init__(
        self,
        R: float,
        p_s: NDArray[np.float64],
        chi_s: float,
        p_e: NDArray[np.float64],
        chi_e: float,
    ):
        self.R = R
        self.p_s = p_s
        self.chi_s = chi_s
        self.p_e = p_e
        self.chi_e = chi_e

        (
            self.c_s,
            self.c_e,
            self.lambda_s,
            self.lambda_e,
            self.z1,
            self.z2,
            self.z3,
            self.q1,
            self.q3,
            first_arc,
            second_arc,
            straight
        ) = get_dubins_parameters(
            p_s, chi_s, p_e, chi_e, R
        )

        # Ensure q1 points from z1 toward z2.
        self.q1 = (
            self.z2 - self.z1
        ) / np.linalg.norm(self.z2 - self.z1)

        self.path_state = "start_wait_negative"
        self.done = False

        self.segment_tolerance = 1e-6

        self.zero_start_turn = (
            np.linalg.norm(self.z1 - self.p_s)
            <= self.segment_tolerance
        )

        self.zero_end_turn = (
            np.linalg.norm(self.z3 - self.z2)
            <= self.segment_tolerance
        )

        if self.zero_start_turn:
            self.path_state = "straight"
        else:
            self.path_state = "start_wait_negative"

    def turn_input(
        self,
        vehicle: SimpleDynamics,
        direction: float,
    ) -> NDArray[np.float64]:
        acceleration = vehicle.v0**2 / (self.R)

        if acceleration > vehicle.max_acceleration + 1e-10:
            raise ValueError(
                "The requested turn radius is smaller than "
                "the vehicle's minimum turn radius."
            )

        return np.array([direction * acceleration])

    def line_input(
        self,
        vehicle: SimpleDynamics,
    ) -> NDArray[np.float64]:
        p = vehicle.x[:2]
        heading = vehicle.x[2]

        line_heading = np.arctan2(
            self.q1[1],
            self.q1[0],
        )

        # Signed cross-track error: positive means left of line.
        left_normal = np.array([
            -self.q1[1],
            self.q1[0],
        ])
        cross_track_error = np.dot(
            p - self.z1,
            left_normal,
        )

        k_path = 0.01
        desired_heading = (
            line_heading
            - np.arctan(k_path * cross_track_error)
        )

        heading_error = wrap_angle(
            desired_heading - heading
        )

        k_heading = 3.0
        acceleration = np.clip(
            k_heading * vehicle.v0 * heading_error,
            -vehicle.max_acceleration,
            vehicle.max_acceleration,
        )

        return np.array([acceleration])

    def update(
        self,
        vehicle: SimpleDynamics,
    ) -> NDArray[np.float64]:
        p = vehicle.x[:2]

        # The loop allows an immediate change of control after
        # crossing a switching half-plane.
        while True:
            if self.path_state == "start_wait_negative":
                if in_half_plane(p, self.z1, -self.q1):
                    self.path_state = "start_wait_positive"
                    continue

                return self.turn_input(
                    vehicle,
                    self.lambda_s,
                )

            if self.path_state == "start_wait_positive":
                if in_half_plane(p, self.z1, self.q1):
                    self.path_state = "straight"
                    continue

                return self.turn_input(
                    vehicle,
                    self.lambda_s,
                )

            if self.path_state == "straight":
                # At both z1 and z2, the switching-plane
                # normal is q1; equivalently, q2 = q1.
                if in_half_plane(p, self.z2, self.q1):
                    if self.zero_end_turn:
                        self.path_state = "complete"
                        self.done = True
                        return np.array([0.0])
                    
                    self.path_state = "end_wait_negative"
                    continue

                return self.line_input(vehicle)

            if self.path_state == "end_wait_negative":
                if in_half_plane(p, self.z3, -self.q3):
                    self.path_state = "end_wait_positive"
                    continue

                return self.turn_input(
                    vehicle,
                    self.lambda_e,
                )

            if self.path_state == "end_wait_positive":
                if in_half_plane(p, self.z3, self.q3):
                    self.path_state = "complete"
                    self.done = True
                    continue

                return self.turn_input(
                    vehicle,
                    self.lambda_e,
                )

            if self.path_state == "complete":
                return np.array([0.0])

    



def in_half_plane(
    position: NDArray[np.float64],
    point: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> bool:
    return np.dot(position - point, normal) >= 0.0

def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def dubins_state_at_time(x0, xtf, t, v, max_acceleration):
    R = v**2/max_acceleration
    c_s, c_e, lambda_s, lambda_e, z1, z2, z3, q1, q3, first_arc, second_arc, straight = get_dubins_parameters(x0[:2], x0[2], xtf[:2], xtf[2], R)
    start_time = R * first_arc / v
    straight_time = straight / v
    end_time = R * second_arc / v
    total_time = start_time + straight_time + end_time

    t = np.clip(t, 0, total_time)

    if t <= start_time:
        heading_change = lambda_s * v * t / R
        position = c_s + rotation_matrix(heading_change) @ (x0[:2] - c_s)
        heading = x0[2] + heading_change

    elif t <= start_time + straight_time:
        segment_time = t - start_time
        position = z1 + v * segment_time * q1
        heading = np.arctan2(q1[1], q1[0])
    else:
        segment_time = t - start_time - straight_time
        heading_change = lambda_e * v * segment_time / R
        position = c_e + rotation_matrix(heading_change) @ (z2 - c_e)
        tangent_heading = np.arctan2(q1[1], q1[0])
        heading = tangent_heading + heading_change

    heading = np.array([wrap_angle(heading)])
    x = np.concat((position, heading))

    return x
