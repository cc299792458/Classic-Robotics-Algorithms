import numpy as np
import matplotlib.pyplot as plt

from utils.dynamic_utils import SecondOrderUnicycle
from advanced_algos.motion_planning.sampling_methods.prm import SS

# Configuration space size (3m x 4m), velocity range [0, 0.036], and angle range [0, 2π]
config_space_width = 3.0
config_space_height = 4.0
velocity_range = (0, 0.036)
angle_range = (0, 2 * np.pi)

# Define control space bounds: linear velocity and angular velocity ranges
angular_velocity_range = (-1.0, 1.0)  # Angular velocity range in radians per second

# Obstacle generation
def generate_obstacles(num_obstacles, space_width, space_height, radius_range):
    """
    Generate random circular obstacles within the configuration space.

    Params:
    - num_obstacles (int): Number of obstacles to generate.
    - space_width (float): Width of the configuration space.
    - space_height (float): Height of the configuration space.
    - radius_range (tuple): Min and max radius for the obstacles.

    Returns:
    - list of dict: Each obstacle represented as a dict with 'x', 'y', and 'radius'.
    """
    obstacles = []
    for _ in range(num_obstacles):
        x = np.random.uniform(0, space_width)
        y = np.random.uniform(0, space_height)
        radius = np.random.uniform(radius_range[0], radius_range[1])
        obstacles.append({'x': x, 'y': y, 'radius': radius})
    return obstacles

def plot_obstacles(obstacles):
    """
    Plot obstacles in the configuration space.
    
    Params:
    - obstacles (list of dict): List of obstacles with 'x', 'y', and 'radius'.
    """
    for obstacle in obstacles:
        circle = plt.Circle((obstacle['x'], obstacle['y']), obstacle['radius'], color='red', fill=True)
        plt.gca().add_patch(circle)

def interpolate_states(state1, state2, num_samples=10):
    """
    Linearly interpolate between two states.

    Params:
    - state1 (np.array): Starting state [x, y, v, theta].
    - state2 (np.array): Ending state [x, y, v, theta].
    - num_samples (int): Number of samples between the two states.

    Returns:
    - list of np.array: List of interpolated states.
    """
    return [state1 + (state2 - state1) * i / (num_samples - 1) for i in range(num_samples)]

def is_collision(state1, state2, obstacles):
    """
    Check if the path between two states collides with any obstacles.
    
    Params:
    - state1 (np.array): The starting state [x, y, v, theta].
    - state2 (np.array): The ending state [x, y, v, theta].
    - obstacles (list of dict): List of obstacles with 'x', 'y', and 'radius'.
    
    Returns:
    - bool: True if any of the interpolated states collide with obstacles, False otherwise.
    """
    # Interpolate between the two states
    interpolated_states = interpolate_states(state1, state2)

    # Check for collisions at each interpolated state
    for state in interpolated_states:
        x, y, _, _ = state
        for obstacle in obstacles:
            dist = np.sqrt((x - obstacle['x'])**2 + (y - obstacle['y'])**2)
            if dist < obstacle['radius']:
                return True
    return False

def plan_with_SS(start, goal, obstacles):
    """
    Plan a path from start to goal using the SS method and SecondOrderUnicycle dynamics.
    
    Params:
    - start (np.array): Start state [x, y, v, theta].
    - goal (np.array): Goal state [x, y, v, theta].
    - obstacles (list of dict): List of obstacles in the configuration space.
    
    Returns:
    - path (list of np.array): The planned path as a list of states.
    """
    # Define the control space bounds (velocity and angular velocity)
    control_space_bounds = (velocity_range, angular_velocity_range)

    # Initialize the planner with the configuration space, dynamic model, and max control duration
    planner = SS(
        start_state=start,
        goal_state=goal,
        state_space_bounds=((0, 0, 0, 0), (config_space_width, config_space_height, 0.036, 2 * np.pi)),
        control_space_bounds=control_space_bounds,
        dynamics=SecondOrderUnicycle(),
        collision_checker=lambda state1, state2: not is_collision(state1, state2, obstacles),
        dt=0.01,  # Updated time step
        control_duration_max=6.0,  # Max control duration set to 6
    )

    # Plan the path
    path = planner.plan()

    return path

if __name__ == '__main__':
    # Define the start and goal states in the format [x, y, velocity, theta]
    start_state = np.array([0.1, 0.1, 0.0, 0.0])
    goal_state = np.array([2.9, 3.9, 0.0, np.pi])

    # Generate obstacles
    num_obstacles = 5
    radius_range = (0.35, 0.4)
    obstacles = generate_obstacles(num_obstacles, config_space_width, config_space_height, radius_range)

    # Plan the path using SS
    path = plan_with_SS(start_state, goal_state, obstacles)

    # Plot the configuration space, obstacles, and the planned path
    plt.figure(figsize=(8, 6))
    plt.xlim(0, config_space_width)
    plt.ylim(0, config_space_height)
    plot_obstacles(obstacles)

    # Plot the planned path
    if path:
        x_path = [state[0] for state in path]
        y_path = [state[1] for state in path]
        plt.plot(x_path, y_path, label="Planned Path", color='blue')
        plt.scatter([start_state[0]], [start_state[1]], color='green', label="Start")
        plt.scatter([goal_state[0]], [goal_state[1]], color='orange', label="Goal")
    else:
        print("No path found!")

    plt.legend()
    plt.grid(True)
    plt.title("Kinodynamic Planning for Second-Order Unicycle")
    plt.show()
