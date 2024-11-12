"""
    Demo for the FSBAS class:
    - Create a simple path with two dimensions.
    - Define maximum velocity and acceleration.
    - Perform trajectory smoothing and plot the results.
"""

from utils.misc_utils import set_seed
from advanced_algos.motion_planning.smoothing import FSBAS

def collision_checker(state):
    """Dummy collision checker: always returns True (no collisions)."""
    return True

if __name__ == '__main__':
    set_seed()

    path = [
        ([0.0, 0.0], [0.0, 0.0]),
        ([1.0, 2.0], [0.0, 0.0]),
        ([3.0, 3.0], [0.0, 0.0]),
        ([4.0, 0.0], [0.0, 0.0]),
    ]

    vmax = [2.0, 2.0]  # [vmax_x, vmax_y]
    amax = [1.0, 1.0]  # [amax_x, amax_y]

    # Initialize the FSBAS object
    fsbas = FSBAS(path, vmax, amax, collision_checker, max_iterations=10)

    # Perform trajectory smoothing
    fsbas.smooth_path(plot_trajectory=True)