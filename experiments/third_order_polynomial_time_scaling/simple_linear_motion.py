"""
    Use Third-Order Polynomial Time Scaling to generate a set of trajectory points
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from basic_algos.trajectory_generation.point_to_point_trajectories import third_order_polynomial_time_scaling

def generate_trajectory_points(coefficients, duration, num_points=100):
    """
    Generates a list of trajectory points using the polynomial coefficients.

    Args:
        coefficients (list): Coefficients of the polynomial (A, B, C, D).
        duration (float): Total duration of the trajectory.
        num_points (int): Number of points in the trajectory.

    Returns:
        np.ndarray: Array of positions at each time step.
    """
    A, B, C, D = coefficients
    time_steps = np.linspace(0, duration, num_points)
    positions = A * time_steps**3 + B * time_steps**2 + C * time_steps + D
    return positions

def plot_trajectory(coefficients, duration, trajectory_points, num_points=100, save_path=None):
    """
    Plot the position, velocity, and acceleration of the trajectory.
    
    Args:
        coefficients (list): Coefficients of the polynomial (A, B, C, D).
        duration (float): Total duration of the trajectory.
        trajectory_points (np.ndarray): Generated trajectory points.
        num_points (int): Number of points in the trajectory.
        save_path (str, optional): Path to save the plot image. If None, the plot is not saved.
    """
    A, B, C, D = coefficients
    time_steps = np.linspace(0, duration, num_points)
    
    # Calculate position, velocity, and acceleration
    positions = A * time_steps**3 + B * time_steps**2 + C * time_steps + D
    velocities = 3 * A * time_steps**2 + 2 * B * time_steps + C
    accelerations = 6 * A * time_steps + 2 * B
    
    # Plot position
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(time_steps, positions, label='Position')
    plt.scatter(time_steps, trajectory_points, color='red', s=10, label='Generated Points')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.legend()
    
    # Plot velocity
    plt.subplot(1, 3, 2)
    plt.plot(time_steps, velocities, label='Velocity', color='green')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Time')
    plt.legend()
    
    # Plot acceleration
    plt.subplot(1, 3, 3)
    plt.plot(time_steps, accelerations, label='Acceleration', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title('Acceleration vs Time')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as {save_path}")
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    
    start_pos = 0
    end_pos = 10
    duration = 5
    max_vel = 1.0

    # Generate third-order polynomial coefficients
    coefficients, duration = third_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0.0, end_vel=0.0, duration=duration, max_vel=max_vel)
    print(f"Third Order Polynomial Coefficients (A, B, C, D): {coefficients}, Duration {duration}")

    # Generate trajectory points
    trajectory_points = generate_trajectory_points(coefficients, duration)
    
    # Define the save path for the image in the same directory as the script
    save_path = os.path.join(log_dir, 'third_order_trajectory_plot.png')
    
    # Plot the trajectory and save the image
    plot_trajectory(coefficients, duration, trajectory_points, save_path=save_path)
