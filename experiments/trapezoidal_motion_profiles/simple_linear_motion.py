"""
    Use Trapezoidal Motion Profiles to generate a set of trajectory points
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from basic_algos.trajectory_generation.point_to_point_trajectories import trapezoidal_motion_profile

def generate_trajectory_points(params, num_points=100):
    """
    Generates a list of trajectory points using the trapezoidal profile parameters.

    Args:
        params (dict): Parameters of the trapezoidal velocity profile.
        num_points (int): Number of points in the trajectory.

    Returns:
        dict: Contains arrays of time, position, velocity, and acceleration.
    """
    t_acc = params['t_acc']
    t_flat = params['t_flat']
    total_time = params['total_time']
    max_vel = params['max_vel']
    max_acc = params['max_acc']
    direction = params['direction']
    start_pos = params['start_pos']
    
    time_steps = np.linspace(0, total_time, num_points)
    positions = []
    velocities = []
    accelerations = []

    for t in time_steps:
        if t < t_acc:  # Acceleration phase
            pos = start_pos + direction * 0.5 * max_acc * t**2
            vel = direction * max_acc * t
            acc = direction * max_acc
        elif t < t_acc + t_flat:  # Constant velocity phase
            pos = start_pos + direction * (params['d_acc'] + max_vel * (t - t_acc))
            vel = direction * max_vel
            acc = 0.0
        else:  # Deceleration phase
            t_dec = t - t_acc - t_flat
            pos = start_pos + direction * (params['d_acc'] + max_vel * t_flat + max_vel * t_dec - 0.5 * max_acc * t_dec**2)
            vel = direction * (max_vel - max_acc * t_dec)
            acc = -direction * max_acc
        
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(acc)

    return {
        'time': time_steps,
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'acceleration': np.array(accelerations)
    }

def plot_trajectory(trajectory_data, save_path=None):
    """
    Plot the position, velocity, and acceleration of the trajectory.
    
    Args:
        trajectory_data (dict): Contains arrays of time, position, velocity, and acceleration.
        save_path (str, optional): Path to save the plot image. If None, the plot is not saved.
    """
    time_steps = trajectory_data['time']
    positions = trajectory_data['position']
    velocities = trajectory_data['velocity']
    accelerations = trajectory_data['acceleration']
    
    # Plot position
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(time_steps, positions, label='Position')
    plt.scatter(time_steps, positions, color='red', s=10, label='Generated Points')
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
    max_vel = 2
    max_acc = 1

    # Generate trapezoidal motion profile parameters
    params = trapezoidal_motion_profile(start_pos, end_pos, max_vel, max_acc)

    # Print the parameters
    print("Trapezoidal Motion Profile Parameters:")
    print(f"Total Time: {params['total_time']}")
    print(f"Acceleration Time (t_acc): {params['t_acc']}")
    print(f"Constant Velocity Time (t_flat): {params['t_flat']}")
    print(f"Maximum Velocity (max_vel): {params['max_vel']}")
    print(f"Maximum Acceleration (max_acc): {params['max_acc']}")
    print(f"Acceleration Distance (d_acc): {params['d_acc']}")
    print(f"Constant Velocity Distance (d_flat): {params['d_flat']}")
    print(f"Direction: {params['direction']}")
    print(f"Start Position: {params['start_pos']}")

    # Generate trajectory points
    trajectory_data = generate_trajectory_points(params)

    # Define the save path for the image in the same directory as the script
    save_path = os.path.join(log_dir, 'trapezoidal_motion_profile_plot.png')

    # Plot the trajectory and save the image
    plot_trajectory(trajectory_data, save_path=save_path)
