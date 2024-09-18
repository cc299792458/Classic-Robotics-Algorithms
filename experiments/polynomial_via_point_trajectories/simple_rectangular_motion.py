import os
import numpy as np
import matplotlib.pyplot as plt

from basic_algos.trajectory_generation.polynomial_via_point_trajectories import polynomial_via_point_trajectories

def plot_multidimensional_trajectory(coefficients, durations, num_points=100, save_path=None):
    """
    Plot the multi-dimensional trajectory.

    Args:
        coefficients (dict): Dictionary of coefficients for each dimension.
        durations (list or np.ndarray): Duration of each trajectory segment.
        num_points (int): Number of points in the trajectory for each segment.
        save_path (str, optional): Path to save the plot image. If None, the plot is not saved.
    """
    # Create a figure for each dimension
    num_dimensions = len(coefficients)
    fig, axes = plt.subplots(num_dimensions, 3, figsize=(12, 4 * num_dimensions))

    # Loop over each dimension
    for dim in range(num_dimensions):
        dim_coeffs = coefficients[dim]

        # Initialize lists to store full trajectories for the current dimension
        full_positions = []
        full_velocities = []
        full_accelerations = []
        full_time_steps = []

        # Compute and concatenate the trajectory segments for the current dimension
        for idx, coeffs in enumerate(dim_coeffs):
            A, B, C, D = coeffs

            # Time steps for this segment
            segment_duration = durations[idx]
            segment_time_steps = np.linspace(0, segment_duration, num_points)

            # Calculate position, velocity, and acceleration for this segment
            pos = A * segment_time_steps**3 + B * segment_time_steps**2 + C * segment_time_steps + D
            vel = 3 * A * segment_time_steps**2 + 2 * B * segment_time_steps + C
            acc = 6 * A * segment_time_steps + 2 * B

            if full_time_steps:
                segment_time_steps += full_time_steps[-1][-1]  # Offset the time steps to align segments
            full_time_steps.append(segment_time_steps)

            full_positions.append(pos)
            full_velocities.append(vel)
            full_accelerations.append(acc)

        # Concatenate the segments to form the full trajectory
        full_time_steps = np.concatenate(full_time_steps)
        full_positions = np.concatenate(full_positions)
        full_velocities = np.concatenate(full_velocities)
        full_accelerations = np.concatenate(full_accelerations)

        # Plot each dimension's trajectory
        axes[dim, 0].plot(full_time_steps, full_positions, label=f'Dimension {dim} - Position')
        axes[dim, 1].plot(full_time_steps, full_velocities, label=f'Dimension {dim} - Velocity', color='green')
        axes[dim, 2].plot(full_time_steps, full_accelerations, label=f'Dimension {dim} - Acceleration', color='orange')

        # Set labels and titles
        axes[dim, 0].set_xlabel('Time')
        axes[dim, 0].set_ylabel('Position')
        axes[dim, 0].set_title(f'Dimension {dim} - Position vs Time')
        axes[dim, 1].set_xlabel('Time')
        axes[dim, 1].set_ylabel('Velocity')
        axes[dim, 1].set_title(f'Dimension {dim} - Velocity vs Time')
        axes[dim, 2].set_xlabel('Time')
        axes[dim, 2].set_ylabel('Acceleration')
        axes[dim, 2].set_title(f'Dimension {dim} - Acceleration vs Time')

        # Add legends
        axes[dim, 0].legend()
        axes[dim, 1].legend()
        axes[dim, 2].legend()

    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as {save_path}")
    
    # Show the plot
    plt.show()