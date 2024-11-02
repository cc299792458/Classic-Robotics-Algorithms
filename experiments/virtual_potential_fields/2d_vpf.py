import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from basic_algos.motion_planning.virtual_potential_fields import VirtualPotentialField

def plot_path(positions, start, goal, obstacles, obstacle_radii):
    """
    Plot the computed path along with the start, goal, and obstacles with areas.
    """
    positions = np.array(positions)
    plt.plot(positions[:, 0], positions[:, 1], color="orange", linewidth=2, linestyle="--", label="Path")
    plt.scatter(start[0], start[1], marker="s", color="green", s=100, label="Start")
    plt.scatter(goal[0], goal[1], marker="*", color="red", s=150, label="Goal")
    for obs, radius in zip(obstacles, obstacle_radii):
        circle = plt.Circle(obs, radius, color="blue", alpha=0.3, edgecolor="black", linewidth=1.5)
        plt.gca().add_patch(circle)
        plt.scatter(obs[0], obs[1], marker="o", color="blue", s=50)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path with Obstacles")
    plt.legend()
    plt.axis("equal")

def plot_potential_field(ax, X, Y, Z):
    """
    Plot 3D potential field surface.
    """
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor="none", alpha=0.9)
    ax.set_title("3D Potential Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Potential")

def plot_contour(X, Y, Z, goal, obstacles):
    """
    Plot contour map of the potential field.
    """
    plt.contour(X, Y, Z, levels=20, cmap="inferno")
    plt.scatter(goal[0], goal[1], marker="*", color="red", s=100, label="Goal")
    for obs in obstacles:
        plt.scatter(obs[0], obs[1], color="blue", s=50)
    plt.title("Potential Field Contour Map")
    plt.legend()

def plot_vector_field(X, Y, Z, goal, obstacles):
    """
    Plot vector field representing the force directions.
    """
    grad_x, grad_y = np.gradient(-Z)
    plt.quiver(X, Y, grad_x, grad_y, color="black", alpha=0.6, scale=50)
    plt.scatter(goal[0], goal[1], marker="*", color="red", s=100, label="Goal")
    for obs in obstacles:
        plt.scatter(obs[0], obs[1], color="blue", s=50)
    plt.title("Force Vector Field")
    plt.legend()

if __name__ == '__main__':
    goal = np.array([8.0, 8.0])
    obstacles = [np.array([4.0, 2.0]), np.array([5.0, 7.0])]
    obstacle_radii = [1.0, 1.0]
    start = np.array([0.0, 0.0])

    # Initialize and run the VPF algorithm with a maximum repulsive force limit
    vpf = VirtualPotentialField(goal, obstacles, obstacle_radii, max_repulsive_force=100.0)
    path = vpf.plan_path(start)

    # Define grid for potential field calculations
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * vpf.k_att * ((X - goal[0]) ** 2 + (Y - goal[1]) ** 2)

    for obs, radius in zip(obstacles, obstacle_radii):
        dist = np.sqrt((X - obs[0]) ** 2 + (Y - obs[1]) ** 2)
        raw_repulsive = np.where(dist <= vpf.d_safe, 0.5 * vpf.k_rep * (1 / dist - 1 / vpf.d_safe) ** 2, 0)
        
        # Apply maximum repulsive force limit
        capped_repulsive = np.clip(raw_repulsive, 0, vpf.max_repulsive_force)
        Z += capped_repulsive

    # 4 Subplots in a single figure
    fig = plt.figure(figsize=(12, 12))
    plt.suptitle("Virtual Potential Field Visualization", fontsize=16)

    # Subplot 1: Path with Obstacles
    ax1 = fig.add_subplot(2, 2, 1)
    plot_path(path, start, goal, obstacles, obstacle_radii)

    # Subplot 2: 3D Potential Field
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_potential_field(ax2, X, Y, Z)

    # Subplot 3: Contour Map
    plt.subplot(2, 2, 3)
    plot_contour(X, Y, Z, goal, obstacles)

    # Subplot 4: Force Vector Field
    plt.subplot(2, 2, 4)
    plot_vector_field(X, Y, Z, goal, obstacles)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
