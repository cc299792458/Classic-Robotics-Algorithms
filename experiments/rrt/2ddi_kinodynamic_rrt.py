import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.misc_utils import set_seed
from utils.dynamic_utils import TwoDimensionalDoubleIntegrator
from advanced_algos.motion_planning.sampling_methods.rrt import KinodynamicRRT

def is_obstacle_free_trajectory(state1, state2, obstacles):
    """
    Check if the trajectory from state1 to state2 is collision-free.
    The trajectory is discretized into multiple points for collision checking.
    """
    num_checks = 10
    s_values = np.linspace(0, 1, num_checks)
    for s in s_values:
        interp_state = state1 + s * (state2 - state1)
        x, y = interp_state[:2]
        for (ox, oy, width, height) in obstacles:
            if ox <= x <= ox + width and oy <= y <= oy + height:
                return False  # Collision detected
    return True  # No collision detected

if __name__ == '__main__':
    set_seed()

    # Define obstacles (list of tuples representing (x, y, width, height))
    obstacles = [
        (30, 30, 20, 20),  # Obstacle 1
        (60, 60, 15, 30)   # Obstacle 2
    ]

    # Define a wrapper function for obstacle checking
    def obstacle_check(state1, state2):
        return is_obstacle_free_trajectory(state1, state2, obstacles)

    # Start and goal states (position and velocity)
    start = [2, 2, 0, 0]
    goal = [8, 8, 0, 0]
    state_limits = ((0, 10), (0, 10), (-2, 2), (-2, 2))
    u_set = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    control_duration = 0.25
    dt = 0.01
    goal_threshold = 0.5

    # Initialize dynamics model
    dynamics_model = TwoDimensionalDoubleIntegrator()

    # Create an instance of KinodynamicRRT
    rrt = KinodynamicRRT(
        start=start,
        goal=goal,
        obstacle_free=obstacle_check,
        max_iters=1_000,
        state_limits=state_limits,
        u_set=u_set,
        dynamics_model=dynamics_model,
        control_duration=control_duration,
        dt=dt,
        goal_threshold=goal_threshold,
    )

    # Run the Kinodynamic RRT algorithm
    path = rrt.plan()

    # Find the closest node to the goal in the tree
    closest_node = rrt.nearest(rrt.goal)
    closest_distance = np.linalg.norm(closest_node[:len(rrt.goal)] - rrt.goal)

    # Print the closest node and its distance to the goal
    print("Closest node to the goal:", closest_node)
    print("Distance to the goal:", closest_distance)

    # Visualization settings
    animate_plot = False  # Set to False for a static plot

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the obstacles (rectangles)
    for (ox, oy, width, height) in obstacles:
        ax.add_patch(
            plt.Rectangle((ox, oy), width, height, color='gray', alpha=0.8)
        )

    # Plot start and goal points
    ax.scatter(start[0], start[1], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')

    ax.set_title('Kinodynamic RRT with Obstacles')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    edges = rrt.all_edges  # List of edges in the order they were added

    if animate_plot:
        # Animation mode
        total_frames = len(edges)

        # Create a function to update the plot
        def update(num):
            ax.clear()

            # Re-plot the obstacles
            for (ox, oy, width, height) in obstacles:
                ax.add_patch(
                    plt.Rectangle((ox, oy), width, height, color='gray', alpha=0.8)
                )

            # Re-plot start and goal points
            ax.scatter(start[0], start[1], color='green', s=100, label='Start')
            ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')

            # Plot the edges up to the current frame
            for edge in edges[:num]:
                p1, p2 = edge
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color='yellow', linestyle='-', linewidth=1, alpha=0.8
                )

            # Plot the path if found and at the last frame
            if path is not None and len(path) > 0 and num == total_frames - 1:
                path_array = np.array(path)
                ax.plot(
                    path_array[:, 0], path_array[:, 1],
                    color='blue', linestyle='-', linewidth=2, label='Path'
                )

            ax.set_title('Kinodynamic RRT with Obstacles: Animated Search Tree')
            ax.legend(loc='upper left')
            ax.grid(True)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=total_frames, interval=1, repeat=False
        )

        plt.show()

    else:
        # Static plot mode
        # Plot all edges
        for edge in edges:
            p1, p2 = edge
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='yellow', linestyle='-', linewidth=1, alpha=0.8)

        # Plot the path if found
        if path is not None and len(path) > 0:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], color='blue', linestyle='-', linewidth=2, label='Path')

        ax.set_title('Kinodynamic RRT')
        ax.legend(loc='upper left')
        plt.show()
