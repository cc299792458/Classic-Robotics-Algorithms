"""
    Demo for the FSBAS class:
    - Create a simple path with two dimensions.
    - Define maximum velocity and acceleration.
    - Perform trajectory smoothing and plot the results.
"""

import numpy as np

from utils.misc_utils import set_seed
from advanced_algos.motion_planning.smoothing import FSBAS

def collision_checker(state, obstacles):
    """
    Check if the given state is in collision with any obstacles.

    Args:
        state (np.ndarray): State as a concatenated position and velocity vector.
                            Shape: (2 * n_dimensions,)
        obstacles (list of tuples): List of obstacles defined as (x, y, width, height).

    Returns:
        bool: True if in collision, False otherwise.
    """
    position = state[0]
    x, y = position
    for (ox, oy, width, height) in obstacles:
        if ox <= x <= ox + width and oy <= y <= oy + height:
            return False  # Collision detected
    return True  # No collision

if __name__ == '__main__':
    set_seed()

    obstacles = [
        (1, 1, 0.5, 0.5),      # Obstacle 1
        (2, 0.0, 0.5, 0.5)     # Obstacle 2
    ]

    path = np.array([
        ([0.0, 0.0], [0.0, 0.0]),
        ([1.0, 2.0], [0.0, 0.0]),
        ([3.0, 3.0], [0.0, 0.0]),
        ([4.0, 0.0], [0.0, 0.0]),
    ])

    vmax = [2.0, 2.0]  # [vmax_x, vmax_y]
    amax = [1.0, 1.0]  # [amax_x, amax_y]

    # Initialize the FSBAS object
    fsbas = FSBAS(path, vmax, amax, lambda state: collision_checker(state, obstacles), max_iterations=100, obstacles=obstacles)

    # Perform trajectory smoothing
    fsbas.smooth_path(plot_trajectory=True)