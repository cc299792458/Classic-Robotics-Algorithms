import os
import numpy as np
import matplotlib.pyplot as plt

from basic_algos.trajectory_generation.polynomial_via_point_trajectories import polynomial_via_point_trajectories

def plot_multidimensional_trajectory(coefficients, durations, waypoints=None, velocities=None, num_points=100, save_path=None):
    """
    Plot the multi-dimensional trajectory.

    Args:
        coefficients (dict): Dictionary of coefficients for each dimension.
        durations (list or np.ndarray): Duration of each trajectory segment.
        waypoints (np.ndarray, optional): Array of waypoints, shape (num_points, num_dimensions).
        velocities (np.ndarray, optional): Array of velocities at each waypoint, shape (num_points, num_dimensions).
        num_points (int): Number of points in the trajectory for each segment.
        save_path (str, optional): Path to save the plot image. If None, the plot is not saved.
    """
    # Create a figure for each dimension
    num_dimensions = len(coefficients)
    fig, axes = plt.subplots(num_dimensions, 3, figsize=(12, 4 * num_dimensions))

    # Initialize lists to store the full 2D trajectory
    full_2d_positions = []

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

        # Add positions to the 2D trajectory if it's a 2D plot
        if num_dimensions == 2:
            full_2d_positions.append(full_positions)

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

    # If it's a 2D trajectory, plot the path in 2D space
    if num_dimensions == 2:
        plt.figure()
        plt.plot(full_2d_positions[0], full_2d_positions[1], label='2D Trajectory')
        
        # Plot waypoints as solid points
        if waypoints is not None:
            plt.scatter(waypoints[:, 0], waypoints[:, 1], color='red', label='Waypoints')

            # Plot velocity vectors at waypoints
            if velocities is not None:
                for i in range(len(waypoints)):
                    if np.linalg.norm(velocities[i]) > 0:  # Only draw if velocity is not zero
                        # Normalize velocity vector to a fixed length
                        fixed_length = 0.2  # Adjust this length as needed
                        vel_norm = velocities[i] / np.linalg.norm(velocities[i]) * fixed_length
                        plt.arrow(waypoints[i, 0], waypoints[i, 1], vel_norm[0], vel_norm[1], 
                                  head_width=0.05, head_length=0.05, fc='gray', ec='gray', label='Velocity' if i == 0 else "")

        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')
        plt.title('2D Trajectory in Space')
        plt.legend()
        plt.axis('equal')
        
        # Save the 2D plot if a save path is provided
        if save_path:
            plt.savefig(save_path.replace('.png', '_2D.png'), dpi=300)
            print(f"2D Trajectory plot saved as {save_path.replace('.png', '_2D.png')}")
        
        plt.show()

if __name__ == '__main__':
    # Set up the example
    waypoints = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])  # Example for 2D
    velocities = np.array([[0, 0], [1, 0], [0, -1], [0, 0]])  # Initial and final velocities for each waypoint
    durations = np.array([1.0, 1.0, 1.0])

    # Generate coefficients for the polynomial trajectories
    coefficients = polynomial_via_point_trajectories(waypoints, velocities, durations)

    # Plot the trajectories and save the plot
    log_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(log_dir, '2D_rectangular_trajectory_plot.png')
    plot_multidimensional_trajectory(coefficients, durations, waypoints=waypoints, velocities=velocities, save_path=save_path)
