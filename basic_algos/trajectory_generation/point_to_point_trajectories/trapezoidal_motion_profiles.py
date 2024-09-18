def trapezoidal_motion_profile(start_pos, end_pos, max_vel, max_acc):
    """
    Generates the parameters for a trapezoidal velocity profile.
    
    Args:
        start_pos (float): Initial position.
        end_pos (float): Final position.
        max_vel (float): Maximum velocity.
        max_acc (float): Maximum acceleration.
        
    Returns:
        dict: Contains key parameters of the trapezoidal profile.
    """
    # Calculate the distance to travel
    distance = abs(end_pos - start_pos)
    direction = 1 if end_pos > start_pos else -1

    # Time to reach maximum velocity
    t_acc = max_vel / max_acc
    
    # Distance covered during acceleration phase
    d_acc = 0.5 * max_acc * t_acc**2

    # Check if the profile is triangular (does not reach max velocity)
    if 2 * d_acc > distance:
        # Triangular profile: Recalculate time to reach peak velocity
        t_acc = (distance / max_acc)**0.5
        max_vel = max_acc * t_acc
        t_flat = 0
    else:
        # Trapezoidal profile
        d_flat = distance - 2 * d_acc
        t_flat = d_flat / max_vel

    # Total motion time
    total_time = 2 * t_acc + t_flat

    return {
        'total_time': total_time,
        't_acc': t_acc,
        't_flat': t_flat,
        'max_vel': max_vel,
        'max_acc': max_acc,
        'd_acc': d_acc,
        'd_flat': d_flat,
        'direction': direction,
        'start_pos': start_pos
    }

if __name__ == '__main__':
    start_pos = 0
    end_pos = 10
    max_vel = 2
    max_acc = 1

    # Get the parameters of the trapezoidal velocity profile
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