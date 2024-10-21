import numpy as np
from scipy.linalg import expm

from utils.math_utils import skew_symmetric

def compute_screw_axis(p, omega):
    """
    Computes the screw axis given a point p on the axis and the angular velocity vector omega.
    """
    v = np.cross(-omega, p)  # Linear velocity part
    return np.concatenate((omega, v))

def screw_twist_matrix(screw):
    """
    Constructs the 4x4 se(3) twist matrix from a 6D screw axis.
    
    Parameters:
    - screw: 6D vector representing the screw axis [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    
    Returns:
    - A 4x4 se(3) matrix.
    """
    screw_twist = np.zeros((4, 4))
    screw_twist[:3, :3] = skew_symmetric(screw[:3])  # Angular part (omega)
    screw_twist[:3, 3] = screw[3:]  # Linear part (v)
    return screw_twist

def forward_kinematics_in_space(M, screws, thetas):
    """
    Computes forward kinematics in the space frame.

    Parameters:
    - M: The home configuration of the end-effector (4x4 homogeneous transformation matrix).
    - screws: A list of joint screw axes in the space frame (6D vectors).
    - thetas: A list of joint angles (in radians).

    Returns:
    - The transformation matrix of the end-effector.
    """
    T = np.eye(4)  # Initialize as identity matrix
    for i in range(len(screws)):
        # Construct the se(3) twist matrix and compute the matrix exponential
        screw_twist = screw_twist_matrix(screws[i])
        T = T @ expm(screw_twist * thetas[i])  # Directly use expm to compute matrix exponential
    
    # Multiply by the home configuration matrix
    return T @ M

def forward_kinematics_in_body(M, screws, thetas):
    """
    Computes forward kinematics in the body frame.

    Parameters:
    - M: The home configuration of the end-effector (4x4 homogeneous transformation matrix).
    - screws: A list of joint screw axes in the body frame (6D vectors).
    - thetas: A list of joint angles (in radians).

    Returns:
    - The transformation matrix of the end-effector.
    """
    T = M  # Start with the home position of the end-effector
    for i in range(len(screws)):
        # Construct the se(3) twist matrix and compute the matrix exponential
        screw_twist = screw_twist_matrix(screws[i])
        T = T @ expm(screw_twist * thetas[i])  # Directly use expm to compute matrix exponential
    
    return T