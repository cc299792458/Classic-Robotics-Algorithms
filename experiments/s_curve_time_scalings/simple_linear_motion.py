import os
import numpy as np
import matplotlib.pyplot as plt

def s_curve_time_scalings(start_pos, end_pos, max_vel, max_acc, jerk):
    """
    Compute the S-curve time scaling parameters for the motion.

    Args:
        start_pos (float): Starting position.
        end_pos (float): Ending position.
        max_vel (float): Maximum velocity.
        max_acc (float): Maximum acceleration.
        jerk (float): The constant jerk (rate of change of acceleration).

    Returns:
        dict: A dictionary containing the time and distances for different phases of the S-curve profile.
    """
    # Calculate the total distance
    total_distance = abs(end_pos - start_pos)
    
    # Time to reach max acceleration with jerk, and distance and velocity when reaching max acceleration
    t_jerk = max_acc / jerk    
    d_jerk = jerk * (t_jerk ** 3) / 6
    v_jerk = jerk * (t_jerk ** 2) / 2

    # Time to reach max velocity with maximum acceleration
    t_acc = (max_vel - v_jerk) / max_acc
    d_acc = v_jerk * t_acc + 0.5 * max_acc * t_acc**2

    # Total distance covered during jerk phases (ramp-up and ramp-down of acceleration)
    d_total_jerk = 4 * d_jerk
    
    # Total distance covered during acceleration and deceleration
    d_total_acc = 2 * d_acc

    if total_distance < d_total_jerk:
        # Case 1: Very Short Distance - Adjust Max Acceleration and Velocity
        max_acc = (6 * total_distance * jerk) ** (1/3)
        t_jerk = max_acc / jerk
        max_vel = jerk * (t_jerk**2) / 2  # Maximum velocity achievable in this profile
        t_acc = 0  # No constant acceleration phase
        d_acc_jerk = (1/6) * jerk * (t_jerk ** 3)
        d_total_acc = 2 * d_acc_jerk
        t_constant_vel = 0
    elif total_distance < d_total_acc + d_total_acc:
        # Case 2: Short Distance - Adjust Velocity
        max_vel = (total_distance - d_total_jerk) / (2 * t_jerk)
        t_acc = 0  # No constant acceleration phase
        t_constant_vel = 0
    else:
        # Case 3: Full S-curve Profile - Complete Motion Profile
        d_constant_vel = total_distance - d_total_acc
        t_constant_vel = d_constant_vel / max_vel

    # Compute the times for different phases
    t_total_jerk = 4 * t_jerk
    t_total_acc = 2 * t_acc
    t_total = t_total_jerk + t_total_acc + t_constant_vel  # Total time including constant velocity phase

    return {
        't_jerk': t_jerk,
        't_acc': t_acc,
        't_constant_vel': t_constant_vel,
        't_total': t_total,
        'd_jerk': d_acc_jerk,
        'd_acc': d_acc,
        'd_constant_vel': d_constant_vel if 'd_constant_vel' in locals() else 0,
        'max_vel': max_vel,
        'max_acc': max_acc,
        'jerk': jerk,
        'start_pos': start_pos,
        'end_pos': end_pos
    }

def generate_s_curve_trajectory_points(params, num_points=100):
    """
    Generates a list of trajectory points using the S-curve profile parameters.

    Args:
        params (dict): Parameters of the S-curve velocity profile.
        num_points (int): Number of points in the trajectory.

    Returns:
        dict: Contains arrays of time, position, velocity, and acceleration.
    """
    start_pos = params['start_pos']
    end_pos = params['end_pos']
    total_distance = abs(end_pos - start_pos)
    direction = 1 if end_pos > start_pos else -1

    t_jerk = params['t_jerk']
    t_acc = params['t_acc']
    t_constant_vel = params['t_constant_vel']
    total_time = params['t_total']
    max_vel = params['max_vel']
    max_acc = params['max_acc']
    jerk = params['jerk']
    
    time_steps = np.linspace(0, total_time, num_points)
    positions = []
    velocities = []
    accelerations = []

    for t in time_steps:
        if t < t_jerk:  # Jerk-up phase
            acc = jerk * t
            vel = 0.5 * jerk * t**2
            pos = start_pos + direction * (1/6) * jerk * t**3
        elif t < 2 * t_jerk:  # Constant acceleration phase
            t1 = t - t_jerk
            acc = max_acc
            vel = v_jerk + max_acc * t1
            pos = start_pos + direction * (2 * d_jerk + v_jerk * t1 + 0.5 * max_acc * t1**2)
        elif t < 2 * t_jerk + t_constant_vel:  # Constant velocity phase
            t1 = t - 2 * t_jerk
            acc = 0
            vel = max_vel
            pos = start_pos + direction * (2 * d_jerk + d_acc + max_vel * t1)
        elif t < 3 * t_jerk + t_constant_vel:  # Deceleration phase
            t1 = t - 2 * t_jerk - t_constant_vel
            acc = -jerk * t1
            vel = max_vel - jerk * t1
            pos = start_pos + direction * (2 * d_jerk + d_acc + max_vel * t_constant_vel - (1/6) * jerk * t1**3)
        else:  # Final phase
            acc = 0
            vel = 0
            pos = end_pos
        
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(acc)

    return {
        'time': time_steps,
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'acceleration': np.array(accelerations)
    }

def plot_trajectory(trajectory_data):
    """
    Plot the position, velocity, and acceleration of the trajectory.
    
    Args:
        trajectory_data (dict): Contains arrays of time, position, velocity, and acceleration.
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
    plt.show()

if __name__ == '__main__':
    # Setting parameters
    start_pos = 0
    end_pos = 10
    max_vel = 2
    max_acc = 1
    jerk = 0.5

    # Generate S-curve motion profile parameters
    params = s_curve_time_scalings(start_pos, end_pos, max_vel, max_acc, jerk)

    # Generate trajectory points
    trajectory_data = generate_s_curve_trajectory_points(params)

    # Plot the trajectory
    plot_trajectory(trajectory_data)