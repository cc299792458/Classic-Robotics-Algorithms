import numpy as np
import matplotlib.pyplot as plt

from basic_algos.motion_planning.search_methods import AStar

# Define the grid size
GRID_WIDTH = 20
GRID_HEIGHT = 20


# Define a complex maze-like obstacles
obstacles = [
    # Horizontal barriers creating a maze-like structure
    (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
    (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1),
    (1, 3), (2, 3), (3, 3), (4, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (12, 3), (13, 3), (14, 3), (16, 3),
    (1, 5), (2, 5), (3, 5), (5, 5), (7, 5), (9, 5), (11, 5), (12, 5), (14, 5), (16, 5), (17, 5), (18, 5),
    (1, 7), (3, 7), (4, 7), (6, 7), (8, 7), (9, 7), (10, 7), (12, 7), (14, 7), (16, 7), (18, 7),
    (1, 9), (2, 9), (3, 9), (5, 9), (7, 9), (9, 9), (11, 9), (12, 9), (14, 9), (16, 9), (18, 9),

    # Vertical barriers creating the maze
    (1, 2), (9, 2), (5, 4), (9, 4), (15, 4),
    (1, 6), (5, 6), (9, 6), (15, 6),
    (1, 8), (5, 8), (9, 8), (15, 8),
    (3, 10), (9, 10), (15, 10), (5, 10), (13, 10), (16, 10),
    (7, 11), (2, 12), (4, 12), (8, 12), (10, 12), (12, 12), (14, 12), (16, 12),
    (1, 13), (3, 13), (5, 13), (7, 13), (9, 13), (11, 13), (13, 13), (15, 13),
    (2, 15), (4, 15), (6, 15), (8, 15), (10, 15), (12, 15), (14, 15), (16, 15),
    (1, 17), (3, 17), (5, 17), (7, 17), (9, 17), (11, 17), (13, 17), (15, 17),
    (4, 18), (6, 18), (8, 18), (10, 18), (12, 18), (14, 18), (16, 18),
    (1, 19), (2, 19), (3, 19), (4, 19), (5, 19), (6, 19), (7, 19), (8, 19),
    (10, 19), (11, 19), (12, 19), (13, 19), (14, 19), (15, 19), (16, 19), (17, 19)
]

# Neighbors function: return neighboring grid cells (4-connected grid)
def get_neighbors(state):
    x, y = state
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < GRID_WIDTH - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < GRID_HEIGHT - 1:
        neighbors.append((x, y + 1))
    return neighbors

# Cost function: in this case, every move has a cost of 1
def cost(state1, state2):
    return 1

# Heuristic function: use Manhattan distance to estimate distance to goal
def heuristic(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

# Constraints function: check if the state is not an obstacle
def is_valid(state):
    return state not in obstacles

if __name__ == '__main__':
    # Start and goal positions
    start = (0, 0)
    goal = (19, 19)

    # Create an instance of AStar
    astar = AStar(
        start=start,
        goal=goal,
        neighbors=get_neighbors,
        cost=cost,
        heuristic=heuristic,
        constraints=is_valid
    )

    # Run A* search
    path = astar.search()

    # Visualization
    grid = np.zeros((GRID_WIDTH, GRID_HEIGHT))

    # Mark obstacles
    for (ox, oy) in obstacles:
        grid[oy, ox] = 1  # Mark obstacles as 1

    # Mark path
    if path:
        for (px, py) in path:
            grid[py, px] = 0.5  # Mark the path as 0.5

    # Plotting with a more visually appealing color scheme
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('Blues')  # Use a 'Blues' colormap for visualization
    cax = ax.imshow(grid, cmap=cmap, origin='lower')

    # Mark the start and goal
    ax.scatter(start[0], start[1], color='green', s=100, marker='*', label='Start', zorder=5)
    ax.scatter(goal[0], goal[1], color='red', s=100, marker='*', label='Goal', zorder=5)

    # Enhance the path visualization
    if path:
        path_x = [x for (x, y) in path]
        path_y = [y for (x, y) in path]
        ax.plot(path_x, path_y, color='orange', linewidth=2.5, label='Path', zorder=4)

    # Title and legends
    ax.set_title('A* Pathfinding with Complex Maze', fontsize=15)
    ax.legend(loc='upper left', fontsize=10)

    # Add grid for better maze visualization
    ax.set_xticks(np.arange(-0.5, GRID_WIDTH, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_HEIGHT, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)

    plt.colorbar(cax, ax=ax, label='Occupied/Free Space')

    # Show the plot
    plt.show()

    # Print path if found
    if path:
        print("Path found:")
        print(path)
    else:
        print("No path found.")