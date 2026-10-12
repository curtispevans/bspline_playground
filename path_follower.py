import numpy as np
from numpy.typing import NDArray
import params as P

def rotation_matrix(theta: float) -> NDArray[np.float64]:
    """
    Create a 2D rotation matrix for a given angle.

    Args:
        theta (float): Angle in radians.

    Returns:
        NDArray[np.float64]: 2x2 rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def get_dubins_parameters(p_s: NDArray[np.float64], chi_s: float, p_e: NDArray[np.float64], chi_e: float, R: float) -> tuple[float, float, float, float]:
    """
    Compute the parameters for a Dubins path.

    Args:
        ps (NDArray[np.float64]): Start position [px, py].
        chis (float): Start heading angle in radians.
        pe (NDArray[np.float64]): End position [px, py].
        chi_e (float): End heading angle in radians.
        R (float): Minimum turning radius.

    Returns:
        tuple: A tuple containing the following parameters:
            - c_s (float): Center of the start circle.
            - c_e (float): Center of the end circle.
            - lambda_s (float): Direction of the start circle (+1 for left, -1 for right).
            - lambda_e (float): Direction of the end circle (+1 for left, -1 for right).
    """
    if R <= 0 or not np.isfinite(R):
        raise ValueError("R must be finite and positive")
    if not (np.all(np.isfinite(p_s)) and np.all(np.isfinite(p_e))
            and np.isfinite(chi_s) and np.isfinite(chi_e)):
        raise ValueError("Dubins start/end poses must be finite")
    # Compute the centers of the start and end circles
    c_rs = p_s + R * rotation_matrix(np.pi/2) @ np.array([np.cos(chi_s), np.sin(chi_s)])  
    c_ls = p_s + R * rotation_matrix(-np.pi/2) @ np.array([np.cos(chi_s), np.sin(chi_s)]) 

    c_re = p_e + R * rotation_matrix(np.pi/2) @ np.array([np.cos(chi_e), np.sin(chi_e)])  
    c_le = p_e + R * rotation_matrix(-np.pi/2) @ np.array([np.cos(chi_e), np.sin(chi_e)])

    theta = np.arctan2(c_re[1] - c_rs[1], c_re[0] - c_rs[0])
    first_arc_L1 = mod(2*np.pi + mod(theta - np.pi/2) - mod(chi_s - np.pi/2))
    second_arc_L1 = mod(2*np.pi + mod(chi_e - np.pi/2) - mod(theta - np.pi/2))
    straight_L1 = np.linalg.norm(c_rs - c_re)
    L1 = straight_L1 + R*first_arc_L1 + R*second_arc_L1
    L1_dict = {"first_arc": first_arc_L1, "straight": straight_L1, "second_arc": second_arc_L1}

    L2 = np.inf
    L2_dict = None
    diff = c_le - c_rs
    ell = np.linalg.norm(diff)
    if ell >= 2 * R - 1e-12:
        theta = np.arctan2(diff[1], diff[0])
        tangent_angle = np.arcsin(np.clip(2 * R / ell, -1.0, 1.0))
        theta2 = theta - np.pi / 2 + tangent_angle
        first_arc_L2 = mod(theta2 - (chi_s - np.pi / 2))
        second_arc_L2 = mod(theta2 + np.pi - (chi_e + np.pi / 2))
        straight_L2 = np.sqrt(max(ell**2 - 4 * R**2, 0.0))
        L2 = straight_L2 + R * first_arc_L2 + R * second_arc_L2
        L2_dict = {"first_arc": first_arc_L2, "straight": straight_L2,
                   "second_arc": second_arc_L2}

    L3 = np.inf
    L3_dict = None
    ell = np.linalg.norm(c_re - c_ls)
    if ell >= 2 * R - 1e-12:
        theta = np.arctan2(c_re[1] - c_ls[1], c_re[0] - c_ls[0])
        tangent_angle = np.arccos(np.clip(2 * R / ell, -1.0, 1.0))
        first_arc_L3 = mod((chi_s + np.pi / 2) - (theta + tangent_angle))
        second_arc_L3 = mod((chi_e - np.pi / 2) - (theta + tangent_angle - np.pi))
        straight_L3 = np.sqrt(max(ell**2 - 4 * R**2, 0.0))
        L3 = straight_L3 + R * first_arc_L3 + R * second_arc_L3
        L3_dict = {"first_arc": first_arc_L3, "straight": straight_L3,
                   "second_arc": second_arc_L3}

    theta = np.arctan2(c_le[1] - c_ls[1], c_le[0] - c_ls[0])
    first_arc_L4 = mod(2*np.pi + mod(chi_s + np.pi/2) - mod(theta + np.pi/2))
    second_arc_L4 = mod(2*np.pi + mod(theta + np.pi/2) - mod(chi_e + np.pi/2))
    straight_L4 = np.linalg.norm(c_ls - c_le)
    L4 = straight_L4 + R*first_arc_L4 + R*second_arc_L4
    L4_dict = {"first_arc": first_arc_L4, "straight": straight_L4, "second_arc": second_arc_L4}

    # Determine the minimum length path
    # Handle coincident turn-circle centers before selecting the path, where
    # the external tangent direction is undefined but the path is just an arc.
    if straight_L1 < 1e-12:
        first_arc_L1 = mod(chi_e - chi_s)
        second_arc_L1 = 0.0
        L1 = R * first_arc_L1
        L1_dict = {"first_arc": first_arc_L1, "straight": 0.0,
                   "second_arc": second_arc_L1}
    if straight_L4 < 1e-12:
        first_arc_L4 = mod(chi_s - chi_e)
        second_arc_L4 = 0.0
        L4 = R * first_arc_L4
        L4_dict = {"first_arc": first_arc_L4, "straight": 0.0,
                   "second_arc": second_arc_L4}

    lengths = [L1, L2, L3, L4]
    min_index = np.argmin(lengths)
    if min_index == 0:
        c_s = c_rs
        c_e = c_re
        lambda_s = 1
        lambda_e = 1
        center_distance = np.linalg.norm(c_e - c_s)
        if center_distance < 1e-12:
            q1 = np.array([np.cos(chi_s), np.sin(chi_s)])
            z1, z2 = p_s.copy(), p_e.copy()
        else:
            q1 = (c_e - c_s) / center_distance
            z1 = c_s + R * rotation_matrix(-np.pi/2) @ q1
            z2 = c_e + R * rotation_matrix(-np.pi/2) @ q1
        first_arc = L1_dict["first_arc"]
        second_arc = L1_dict["second_arc"]
        straight = L1_dict["straight"]
    elif min_index == 1:
        c_s = c_rs
        c_e = c_le
        lambda_s = 1
        lambda_e = -1
        ell = np.linalg.norm(c_e - c_s)
        theta = np.arctan2(c_e[1] - c_s[1], c_e[0] - c_s[0])
        theta2 = theta - np.pi/2 + np.arcsin(np.clip(2*R/ell, -1.0, 1.0))
        q1 = rotation_matrix(theta2 + np.pi/2) @ np.array([1, 0])
        z1 = c_s + R * rotation_matrix(theta2) @ np.array([1, 0])
        z2 = c_e + R * rotation_matrix(theta2 + np.pi) @ np.array([1, 0])
        first_arc = L2_dict["first_arc"]
        second_arc = L2_dict["second_arc"]
        straight = L2_dict["straight"]

    elif min_index == 2:
        c_s = c_ls
        c_e = c_re
        lambda_s = -1
        lambda_e = 1
        ell = np.linalg.norm(c_e - c_s)
        theta = np.arctan2(c_e[1] - c_s[1], c_e[0] - c_s[0])
        theta2 = np.arccos(np.clip(2*R/ell, -1.0, 1.0))
        q1 = rotation_matrix(theta + theta2 - np.pi/2) @ np.array([1, 0])
        z1 = c_s + R * rotation_matrix(theta + theta2) @ np.array([1, 0])
        z2 = c_e + R * rotation_matrix(theta + theta2 - np.pi) @ np.array([1, 0])
        first_arc = L3_dict["first_arc"]
        second_arc = L3_dict["second_arc"]
        straight = L3_dict["straight"]
    else:
        c_s = c_ls
        c_e = c_le
        lambda_s = -1
        lambda_e = -1
        center_distance = np.linalg.norm(c_e - c_s)
        if center_distance < 1e-12:
            q1 = np.array([np.cos(chi_s), np.sin(chi_s)])
            z1, z2 = p_s.copy(), p_e.copy()
        else:
            q1 = (c_e - c_s) / center_distance
            z1 = c_s + R * rotation_matrix(np.pi/2) @ q1
            z2 = c_e + R * rotation_matrix(np.pi/2) @ q1
        first_arc = L4_dict["first_arc"]
        second_arc = L4_dict["second_arc"]
        straight = L4_dict["straight"]

    z3 = p_e
    q3 = rotation_matrix(chi_e) @ np.array([1, 0])

    return c_s, c_e, lambda_s, lambda_e, z1, z2, z3, q1, q3, first_arc, second_arc, straight
    



def mod(x: float) -> float:
    return x % (2.0 * np.pi)