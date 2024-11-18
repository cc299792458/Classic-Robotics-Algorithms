import numpy as np

from utils.misc_utils import set_seed
from advanced_algos.motion_planning.sampling_methods.single_query import RampPlanner

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
    position = state[:2]
    x, y = position
    for (ox, oy, width, height) in obstacles:
        if ox <= x <= ox + width and oy <= y <= oy + height:
            return True  # Collision detected
    return False  # No collision


def is_obstacle_free_trajectory(state1, state2, obstacles):
    """
    Check if the trajectory from state1 to state2 is collision-free.
    The trajectory is discretized into multiple points for collision checking.

    Args:
        state1 (np.ndarray): Starting state as a concatenated position and velocity vector.
                             Shape: (2 * n_dimensions,)
        state2 (np.ndarray): Ending state as a concatenated position and velocity vector.
                             Shape: (2 * n_dimensions,)
        obstacles (list of tuples): List of obstacles defined as (x, y, width, height).

    Returns:
        bool: True if the trajectory is collision-free, False otherwise.
    """
    num_checks = 20
    for s in np.linspace(0, 1, num_checks):
        interp_state = state1 + s * (state2 - state1)
        if collision_checker(interp_state, obstacles):
            return False  # Collision detected
    return True  # No collision detected


if __name__ == '__main__':
    # Set random seed for reproducibility
    set_seed()

    # Define obstacles (list of tuples representing (x, y, width, height))
    obstacles = [
        (3, 3, 2, 2),      # Obstacle 1
        (6, 6, 1.5, 3)     # Obstacle 2
    ]

    # Define start and goal states (position and velocity)
    start = np.array([2.0, 2.0, 0.0, 0.0])  # [x, y, vx, vy]
    goal = np.array([8.0, 8.0, -1.0, 1.0])  # [x, y, vx, vy]

    # Define state limits
    position_limits = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))  # [min_pos, max_pos]
    velocity_limits = np.array([2.0, 2.0])

    # Define control limits (acceleration)
    acceleration_limits = np.array([1.0, 1.0])

    # Define maximum iterations
    max_iters = 1000

    # Initialize RampPlanner
    planner = RampPlanner(
        start=start,
        goal=goal,
        max_iters=max_iters,
        collision_checker=lambda state: not collision_checker(state, obstacles),
        position_limits=position_limits,
        vmax=velocity_limits,
        amax=acceleration_limits
    )

    # Run the planner
    path = planner.plan(visualize=True, visualization_args=dict(obstacles=obstacles))

    # Output the results
    if path:
        print("Path found:")
        for idx, state in enumerate(path):
            pos, vel = state[:2], state[2:]
            print(f"Step {idx + 1}: Position = {pos}, Velocity = {vel}")
    else:
        print("No path found within the maximum iterations.")
