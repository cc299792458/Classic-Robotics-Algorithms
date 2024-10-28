import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from utils.math_utils import line_intersects_rect
from basic_algos.motion_planning.sampling_methods.prm import PRM

# Collision checker that uses the above functions
def collision_checker(p1, p2, obstacles):
    """Check if the line segment between p1 and p2 intersects any obstacles."""
    for (ox, oy, width, height) in obstacles:
        rect = ((ox, oy), (ox + width, oy + height))
        if line_intersects_rect(p1, p2, rect):
            return False  # Collision detected
    return True  # No collision

if __name__ == '__main__':
    set_seed()

    # Define obstacles as rectangles: (x, y, width, height)
    obstacles = [
        (30, 30, 20, 20),  # Obstacle 1
        (60, 60, 15, 30),  # Obstacle 2
        (20, 70, 40, 10),  # Obstacle 3
    ]

    # Wrapper for collision checking that includes obstacles
    def is_collision_free(p1, p2):
        return collision_checker(p1, p2, obstacles)

    num_samples = 200
    k_neighbors = 10

    # Create PRM instance
    prm = PRM(
        num_samples=num_samples,
        k_neighbors=k_neighbors,
        collision_checker=is_collision_free,
        sampling_area=((0, 100), (0, 100))
    )

    # Build the roadmap
    prm.construct_roadmap()

    # Define start and goal configurations
    start = (10, 10)
    goal = (90, 90)

    # Find a path
    path = prm.query(start, goal)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for (ox, oy, width, height) in obstacles:
        obstacle = plt.Rectangle((ox, oy), width, height, color='gray')
        ax.add_patch(obstacle)

    # Plot edges in the roadmap
    for (u, v) in prm.roadmap.edges():
        x_values = [u[0], v[0]]
        y_values = [u[1], v[1]]
        ax.plot(x_values, y_values, color='lightblue', linewidth=0.5)

    # Plot nodes
    nodes_x = [node[0] for node in prm.nodes]
    nodes_y = [node[1] for node in prm.nodes]
    ax.scatter(nodes_x, nodes_y, color='black', s=5)

    # Plot start and goal
    ax.scatter(start[0], start[1], color='green', s=100, marker='*', label='Start')
    ax.scatter(goal[0], goal[1], color='red', s=100, marker='*', label='Goal')

    # Plot path
    if path:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        ax.plot(path_x, path_y, color='orange', linewidth=2, label='Path')
    
    ax.text(
            98, 2, f'Num samples: {num_samples}\nK neighbors: {k_neighbors}\nCollision check count: {prm.collision_check_count}',
            fontsize=12, color='black', ha='right'
        )

    ax.set_title('Probabilistic Roadmap (PRM)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.show()