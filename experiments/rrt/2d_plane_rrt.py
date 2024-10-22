import numpy as np
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from basic_algos.motion_planning.sampling_methods.rrt import RRT

def is_obstacle_free(x_nearest, x_new):
    return True  # Always return True (no obstacles)

if __name__ == '__main__':
    set_seed()

    # Creating an RRT object
    start = [0, 0]
    goal = [100, 100]
    rrt = RRT(start=start, goal=goal, obstacle_free=is_obstacle_free, max_iters=1000, delta_distance=5, goal_sample_rate=0.1)

    # Running the RRT algorithm to find a path
    path = rrt.plan()

    # Visualization of the search tree and final path
    plt.figure(figsize=(6, 6))

    # Plot all the search tree edges
    for edge in rrt.all_edges:
        p1, p2 = edge
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y-', alpha=0.5)  # Yellow lines for search tree

    # If a path is found, plot it
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'b-', label='Path')  # Plot the final path in blue

    # Plot start and goal points
    plt.scatter(start[0], start[1], color='g', label='Start')  # Plot the start point
    plt.scatter(goal[0], goal[1], color='r', label='Goal')  # Plot the goal point

    # Labels and title
    plt.title('RRT Search Tree and Path Visualization')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 110)
    plt.ylim(0, 110)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()