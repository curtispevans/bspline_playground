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
    if np.linalg.norm(p_e - p_s) < 2 * R:
        raise ValueError("The distance between start and end points is less than 2*R. Dubins path is not feasible.")

    # Compute the centers of the start and end circles
    c_rs = p_s + R * rotation_matrix(np.pi/2) @ np.array([np.cos(chi_s), np.sin(chi_s)])  
    c_ls = p_s + R * rotation_matrix(-np.pi/2) @ np.array([np.cos(chi_s), np.sin(chi_s)]) 

    c_re = p_e + R * rotation_matrix(np.pi/2) @ np.array([np.cos(chi_e), np.sin(chi_e)])  
    c_le = p_e + R * rotation_matrix(-np.pi/2) @ np.array([np.cos(chi_e), np.sin(chi_e)]) 

    # compute L1, L2, L3, L4 distances
    theta = np.arctan2(c_re[1] - c_rs[1], c_re[0] - c_rs[0])
    L1 = np.linalg.norm(c_rs - c_re) + R*mod(2*np.pi + mod(theta - np.pi/2) - mod(chi_s - np.pi/2)) + R*mod(2*np.pi + mod(chi_e-np.pi/2) - mod(theta - np.pi/2))

    diff = c_le - c_rs
    ell = np.linalg.norm(diff)
    theta = np.arctan2(diff[1], diff[0])
    ell = np.linalg.norm(c_le - c_rs)
    theta2 = theta - np.pi/2 + np.arcsin(2*R/ell)
    L2 = np.sqrt(ell**2 - 4*R**2) + R*mod(2*np.pi + mod(theta2) - mod(chi_s - np.pi/2)) + R*mod(2*np.pi + mod(theta2 + np.pi) - mod(chi_e + np.pi/2))

    theta = np.arctan2(c_re[1] - c_ls[1], c_re[0] - c_ls[0])
    ell = np.linalg.norm(c_re - c_ls)
    theta2 = np.arccos(2*R/ell)
    L3 = np.sqrt(ell**2 - 4*R**2) + R*mod(2*np.pi + mod(chi_s + np.pi/2) - mod(theta + theta2)) + R*mod(2*np.pi + mod(chi_e - np.pi/2) - mod(theta + theta2 - np.pi))

    theta = np.arctan2(c_le[1] - c_ls[1], c_le[0] - c_ls[0])
    L4 = np.linalg.norm(c_ls - c_le) + R*mod(2*np.pi + mod(chi_s + np.pi/2) - mod(theta + np.pi/2)) + R*mod(2*np.pi + mod(theta + np.pi/2) - mod(chi_e + np.pi/2))

    # Determine the minimum length path
    lengths = [L1, L2, L3, L4]
    min_index = np.argmin(lengths)
    if min_index == 0:
        c_s = c_rs
        c_e = c_re
        lambda_s = 1
        lambda_e = 1
        q1 = (c_e - c_s) / np.linalg.norm(c_e - c_s)
        z1 = c_s + R * rotation_matrix(-np.pi/2) @ q1
        z2 = c_e + R * rotation_matrix(-np.pi/2) @ q1
        
    elif min_index == 1:
        c_s = c_rs
        c_e = c_le
        lambda_s = 1
        lambda_e = -1
        ell = np.linalg.norm(c_e - c_s)
        theta = np.arctan2(c_e[1] - c_s[1], c_e[0] - c_s[0])
        theta2 = theta - np.pi/2 + np.arcsin(2*R/ell)
        q1 = rotation_matrix(theta2 + np.pi/2) @ np.array([1, 0])
        z1 = c_s + R * rotation_matrix(theta2) @ np.array([1, 0])
        z2 = c_e + R * rotation_matrix(theta2 + np.pi) @ np.array([1, 0])
    elif min_index == 2:
        c_s = c_ls
        c_e = c_re
        lambda_s = -1
        lambda_e = 1
        ell = np.linalg.norm(c_e - c_s)
        theta = np.arctan2(c_e[1] - c_s[1], c_e[0] - c_s[0])
        theta2 = np.arccos(2*R/ell)
        q1 = rotation_matrix(theta + theta2 - np.pi/2) @ np.array([1, 0])
        z1 = c_s + R * rotation_matrix(theta + theta2) @ np.array([1, 0])
        z2 = c_e + R * rotation_matrix(theta + theta2 - np.pi) @ np.array([1, 0])
    else:
        c_s = c_ls
        c_e = c_le
        lambda_s = -1
        lambda_e = -1
        q1 = (c_e - c_s) / np.linalg.norm(c_e - c_s)    
        z1 = c_s + R * rotation_matrix(np.pi/2) @ q1
        z2 = c_e + R * rotation_matrix(np.pi/2) @ q1

    z3 = p_e
    q3 = rotation_matrix(chi_e) @ np.array([1, 0])

    return c_s, c_e, lambda_s, lambda_e, z1, z2, z3, q1, q3


def mod(x: float) -> float:
    return x % (2.0 * np.pi)