import numpy as np

from utils.misc_utils import set_seed
from advanced_algos.motion_planning.sampling_methods.single_query import RampPlanner

def collision_checker(state, obstacles):
    """
    Check if the given state is in collision with any obstacles.

    Args:
        state (tuple of np.ndarray): State as (position, velocity).
        obstacles (list of tuples): List of obstacles defined as (x, y, width, height).

    Returns:
        bool: True if in collision, False otherwise.
    """
    position, _ = state
    x, y = position[:2]
    for (ox, oy, width, height) in obstacles:
        if ox <= x <= ox + width and oy <= y <= oy + height:
            return True  # Collision detected
    return False  # No collision

def is_obstacle_free_trajectory(state1, state2, obstacles):
    """
    Check if the trajectory from state1 to state2 is collision-free.
    The trajectory is discretized into multiple points for collision checking.

    Args:
        state1 (tuple of np.ndarray): Starting state as (position, velocity).
        state2 (tuple of np.ndarray): Ending state as (position, velocity).
        obstacles (list of tuples): List of obstacles defined as (x, y, width, height).

    Returns:
        bool: True if the trajectory is collision-free, False otherwise.
    """
    num_checks = 20
    for s in np.linspace(0, 1, num_checks):
        interp_pos = state1[0] + s * (state2[0] - state1[0])
        interp_vel = state1[1] + s * (state2[1] - state1[1])
        interp_state = (interp_pos, interp_vel)
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
    start_pos = np.array([2.0, 2.0])
    start_vel = np.array([0.0, 0.0])
    start = (start_pos, start_vel)

    goal_pos = np.array([8.0, 8.0])
    goal_vel = np.array([0.0, 0.0])
    goal = (goal_pos, goal_vel)

    # Define state limits
    state_limits = {
        'position': (np.array([0.0, 0.0]), np.array([10.0, 10.0])),
        'velocity': (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    }

    # Define control limits (acceleration)
    control_limits = {
        'acceleration': (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    }

    # Define maximum iterations
    max_iters = 1000

    # Initialize RampPlanner
    planner = RampPlanner(
        start=start,
        goal=goal,
        max_iters=max_iters,
        collision_checker=lambda state: collision_checker(state, obstacles),
        state_limits=state_limits,
        control_limits=control_limits
    )

    # Run the planner
    path = planner.plan()

    # Output the results
    if path:
        print("Path found:")
        for idx, state in enumerate(path):
            pos, vel = state
            print(f"Step {idx + 1}: Position = {pos}, Velocity = {vel}")
    else:
        print("No path found within the maximum iterations.")
