import numpy as np
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from basic_algos.motion_planning.sampling_methods.rrt import RRT

def is_obstacle_free(x_nearest, x_new, obstacles):
    """Check if the path from x_nearest to x_new intersects with any obstacles."""
    for (ox, oy, width, height) in obstacles:
        if (min(x_nearest[0], x_new[0]) <= ox + width and max(x_nearest[0], x_new[0]) >= ox and
            min(x_nearest[1], x_new[1]) <= oy + height and max(x_nearest[1], x_new[1]) >= oy):
            return False
    return True

if __name__ == '__main__':
    set_seed()

    # Create obstacles (list of tuples representing (x, y, width, height))
    obstacles = [(30, 30, 20, 20), (60, 60, 15, 30)]  # Two rectangular obstacles

    # Define a new wrapper function for obstacle checking
    def obstacle_check(x_nearest, x_new):
        return is_obstacle_free(x_nearest, x_new, obstacles)

    # Start, goal, and create RRT object with the obstacle check function
    start = [10, 10]
    goal = [100, 100]
    rrt_with_obstacles = RRT(start=start, goal=goal, obstacle_free=obstacle_check, max_iters=1000, delta_distance=5, goal_sample_rate=0.1)

    # Run the RRT algorithm with obstacles
    path_with_obstacles = rrt_with_obstacles.plan()

    # Visualization of the search tree, final path, and obstacles
    plt.figure(figsize=(6, 6))

    # Plot the obstacles (rectangles)
    for (ox, oy, width, height) in obstacles:
        plt.gca().add_patch(plt.Rectangle((ox, oy), width, height, color='gray', alpha=0.5))

    # Plot all the search tree edges
    for edge in rrt_with_obstacles.all_edges:
        p1, p2 = edge
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', alpha=0.5)  # Yellow lines for search tree

    # If a path is found, plot it
    if path_with_obstacles is not None and len(path_with_obstacles) > 0:
        path_with_obstacles = np.array(path_with_obstacles)
        plt.plot(path_with_obstacles[:, 0], path_with_obstacles[:, 1], 'b-', label='Path')  # Plot the final path in blue

    # Plot start and goal points
    plt.scatter(start[0], start[1], color='g', label='Start')  # Plot the start point
    plt.scatter(goal[0], goal[1], color='r', label='Goal')  # Plot the goal point

    # Add text to indicate the number of expanded nodes in the top-right corner
    plt.text(105, 105, f'Nodes expanded: {rrt_with_obstacles.num_nodes}', fontsize=12, color='black', ha='right')

    # Labels and title
    plt.title('RRT with Obstacles: Search Tree and Path Visualization')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 110)
    plt.ylim(0, 110)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
