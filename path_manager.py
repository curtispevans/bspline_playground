import numpy as np
from numpy.typing import NDArray
import params as P
from path_follower import get_dubins_parameters, rotation_matrix
from dynamics import SimpleDynamics


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