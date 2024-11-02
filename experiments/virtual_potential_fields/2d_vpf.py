import numpy as np
import matplotlib.pyplot as plt

from basic_algos.motion_planning.virtual_potential_fields import VirtualPotentialField

def plot_path(positions, start, goal, obstacles, obstacle_radii):
    """
    Plot the computed path along with the start, goal, and obstacles with areas.
    
    Parameters:
    - positions: List of positions representing the path.
    - start: Starting position (np.array).
    - goal: Target position (np.array).
    - obstacles: List of obstacle positions (list of np.array).
    - obstacle_radii: List of radii for each obstacle.
    """
    positions = np.array(positions)
    plt.figure(figsize=(8, 8))
    
    # Plot path
    plt.plot(positions[:, 0], positions[:, 1], color="orange", linewidth=2, linestyle="--", label="Path")
    
    # Plot start and goal points with enhanced markers
    plt.scatter(start[0], start[1], marker="s", color="green", s=100, label="Start")  # Start point
    plt.scatter(goal[0], goal[1], marker="*", color="red", s=150, label="Goal")       # Goal point
    
    # Plot obstacles as shaded circles
    for obs, radius in zip(obstacles, obstacle_radii):
        circle = plt.Circle(obs, radius, color="blue", alpha=0.3, edgecolor="black", linewidth=1.5)  # Obstacle area as shaded circle
        plt.gca().add_patch(circle)
        plt.scatter(obs[0], obs[1], marker="o", color="blue", s=50)  # Obstacle center

    # Add grid and labels for better aesthetics
    plt.legend(fontsize=10)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Virtual Potential Field Path Planning with Obstacle Areas", fontsize=14)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Define goal, obstacles, obstacle radii, and starting position
    goal = np.array([8.0, 8.0])
    obstacles = [np.array([4.0, 2.0]), np.array([5.0, 7.0])]  # Adjusted positions to avoid direct path
    obstacle_radii = [1.0, 1.0]  # Radii for obstacles to define their area
    start = np.array([0.0, 0.0])

    # Initialize and run the VPF algorithm with modified obstacles
    vpf = VirtualPotentialField(goal, obstacles, obstacle_radii)
    path = vpf.compute_path(start)

    # Plot the resulting path with obstacles as areas
    plot_path(path, start, goal, obstacles, obstacle_radii)