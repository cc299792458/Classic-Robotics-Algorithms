import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.misc_utils import set_seed
from utils.math_utils import line_intersects_rect
from basic_algos.motion_planning.sampling_methods.rrt import RRTConnect

def is_obstacle_free(x_nearest, x_new, obstacles):
    """Check if the path from x_nearest to x_new intersects with any obstacles."""
    for (ox, oy, width, height) in obstacles:
        obstacle_rect = ((ox, oy), (ox + width, oy + height))
        if line_intersects_rect(x_nearest, x_new, obstacle_rect):
            return False  # The path intersects an obstacle
    return True  # No collision detected

if __name__ == '__main__':
    set_seed()

    # Create obstacles (list of tuples representing (x, y, width, height))
    obstacles = [
        (30, 30, 20, 20),  # Obstacle 1
        (60, 60, 15, 30)   # Obstacle 2
    ]

    # Define a wrapper function for obstacle checking
    def obstacle_check(x_nearest, x_new):
        return is_obstacle_free(x_nearest, x_new, obstacles)

    # Start and goal positions
    start = [20, 20]
    goal = [80, 80]
    sampling_range = ((0, 100), (0, 100))

    # Create an instance of RRTConnect with the obstacle check function
    rrt_connect = RRTConnect(
        start=start,
        goal=goal,
        obstacle_free=obstacle_check,
        max_iters=500,
        delta_distance=5,
        sampling_range=sampling_range
    )

    # Run the RRT-Connect algorithm with obstacles
    start_time = time.time()
    path = rrt_connect.plan()
    end_time = time.time()
    planning_time = end_time - start_time

    # Print the planning time
    print(f"Planning time: {planning_time:.4f} seconds")

    # Visualization of the search trees, final path, and obstacles
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the obstacles (rectangles)
    for (ox, oy, width, height) in obstacles:
        ax.add_patch(
            plt.Rectangle((ox, oy), width, height, color='gray')
        )

    # Plot start and goal points
    ax.scatter(start[0], start[1], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')

    ax.set_title('RRT-Connect')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # List to store the number of nodes up to each frame in the correct order
    num_nodes_list = [2]  # Start with the start and goal nodes
    current_num_nodes = 2

    # Track node count directly from the expansion order in `all_edges`
    for _ in rrt_connect.all_edges:
        current_num_nodes += 1
        num_nodes_list.append(current_num_nodes)

    # Total frames based on the number of edges
    total_frames = len(rrt_connect.all_edges)

    # Create a function to update the plot
    def update(num):
        ax.clear()

        # Re-plot the obstacles
        for (ox, oy, width, height) in obstacles:
            ax.add_patch(
                plt.Rectangle((ox, oy), width, height, color='gray')
            )

        # Re-plot start and goal points
        ax.scatter(start[0], start[1], color='green', s=100, label='Start')
        ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')

        # Plot edges up to the current frame in expansion order
        for edge in rrt_connect.all_edges[:num]:
            p1, p2 = edge
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color='orange' if edge[0] in rrt_connect.parent else 'purple',
                linestyle='-', linewidth=1, alpha=0.8
            )

        # Plot the path if found and at the last frame
        if path is not None and len(path) > 0 and num == total_frames - 1:
            path_array = np.array(path)
            ax.plot(
                path_array[:, 0], path_array[:, 1],
                color='blue', linestyle='-', linewidth=2, label='Path'
            )

        # Dynamic update of the number of expanded nodes
        if num > 0:
            current_num_nodes = num_nodes_list[num - 1]
        else:
            current_num_nodes = 2  # Start and goal nodes only

        # Add text to indicate the number of expanded nodes dynamically
        ax.text(
            105, 105, f'Nodes expanded: {current_num_nodes}',
            fontsize=12, color='black', ha='right'
        )

        ax.set_title('RRT-Connect')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=50, repeat=False
    )

    plt.show()
