import numpy as np

def skew_symmetric(vector):
    """
    Converts a 3D vector into a skew-symmetric matrix.
    """
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])