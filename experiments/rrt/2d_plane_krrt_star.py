import numpy as np
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from utils.math_utils import line_intersects_rect
from basic_algos.motion_planning.sampling_methods.rrt import kRRTStar

# NOTE: Why set max_iters=2000 and there is a sharp corner?
# NOTE: Why rrt* seems faster than rrt?

def is_obstacle_free(p1, p2, obstacles):
    """Check if the line segment between p1 and p2 is collision-free."""
    for (ox, oy, width, height) in obstacles:
        # Correctly construct rect as a tuple of two tuples
        rect = ((ox, oy), (ox + width, oy + height))
        if line_intersects_rect(p1, p2, rect):
            return False  # Collision detected
    return True  # No collision

if __name__ == '__main__':
    set_seed()

    # Define obstacles as rectangles (x, y, width, height)
    obstacles = [
        (30, 30, 20, 20),  # Obstacle 1
        (60, 60, 15, 30),  # Obstacle 2
        (20, 70, 40, 10),  # Obstacle 3
    ]

    # Wrapper for obstacle checking
    def obstacle_check(p1, p2):
        return is_obstacle_free(p1, p2, obstacles)

    # Start and goal positions
    start = (10, 10)
    goal = (90, 90)
    sampling_range = ((0, 100), (0, 100))

    # Create an instance of kRRTStar
    rrt_star = kRRTStar(
        start=start,
        goal=goal,
        obstacle_free=obstacle_check,
        max_iters=10000,
        delta_distance=5.0,
        sampling_range=sampling_range
    )

    # Run the RRT* algorithm
    path = rrt_star.plan()

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for (ox, oy, width, height) in obstacles:
        ax.add_patch(
            plt.Rectangle((ox, oy), width, height, color='gray', alpha=0.8)
        )

    # Plot start and goal points
    ax.scatter(start[0], start[1], color='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='red', s=100, label='Goal')

    ax.set_title('kRRT* Path Planning')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot the edges
    for edge in rrt_star.all_edges:
        p1, p2 = edge
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color='yellow', linestyle='-', linewidth=1.0, alpha=0.8
        )

    # Plot the path if found
    if path:
        path_array = np.array(path)
        ax.plot(
            path_array[:, 0], path_array[:, 1],
            color='blue', linestyle='-', linewidth=2, label='Path'
        )

    plt.show()