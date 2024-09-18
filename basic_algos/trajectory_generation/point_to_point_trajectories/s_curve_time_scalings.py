def s_curve_time_scalings(total_distance, max_vel, max_acc, max_jerk):
    """
    Compute the S-curve time scaling parameters.

    Args:
        total_distance (float): The total distance to be covered.
        max_vel (float): Maximum velocity.
        max_acc (float): Maximum acceleration.
        max_jerk (float): Maximum jerk.

    Returns:
        dict: A dictionary containing the time and distances for different phases of the S-curve profile.
    """
    # Time to reach max acceleration with max jerk
    t_jerk = max_acc / max_jerk

    # Distance and velocity when reaching max acceleration
    d_acc_jerk = max_jerk * t_jerk**3 / 6
    v_acc_jerk = max_jerk * t_jerk**2 / 2

    # Time to reach max velocity with maximum acceleration
    t_acc = (max_vel - v_acc_jerk) / max_acc
    d_acc = v_acc_jerk * t_acc + 0.5 * max_acc * t_acc**2

    # Total distance to reach maximum velocity
    d_total_acc = 2 * (d_acc_jerk + d_acc)

    # Check if we need a triangular profile
    if total_distance < d_total_acc:
        # Triangular S-curve Profile: Cannot reach max velocity and max acceleration
        # Adjust max_acc and max_vel to fit within the total distance
        max_acc = (max_jerk * total_distance) ** (1/3)
        t_jerk = max_acc / max_jerk
        max_vel = max_jerk * t_jerk**2
        t_acc = 0  # No constant acceleration phase

        # Update distances
        d_acc_jerk = max_jerk * t_jerk**3 / 6
        d_total_acc = 2 * d_acc_jerk
        t_constant_vel = 0  # No constant velocity phase
    else:
        # Full or Modified S-curve Profile
        d_flat = total_distance - d_total_acc

        if d_flat > 0:
            # Full S-curve Profile
            t_constant_vel = d_flat / max_vel
        else:
            # Modified S-curve Profile
            max_vel = (total_distance - d_total_acc) / (2 * t_acc + t_jerk)
            t_constant_vel = 0

    # Total motion time
    t_total_acc = 2 * (t_jerk + t_acc)
    total_time = t_total_acc + t_constant_vel

    return {
        'total_time': total_time,
        't_jerk': t_jerk,
        't_acc': t_acc,
        't_constant_vel': t_constant_vel,
        'max_vel': max_vel,
        'max_acc': max_acc,
        'max_jerk': max_jerk,
        'd_acc_jerk': d_acc_jerk,
        'd_total_acc': d_total_acc,
        'd_flat': d_flat if d_flat > 0 else 0
    }

if __name__ == '__main__':
    params = s_curve_time_scalings(total_distance=10, max_vel=2, max_acc=1, max_jerk=0.5)
    print("S-curve Motion Profile Parameters:")
    print(f"Total Time: {params['total_time']}")
    print(f"Jerk Time (t_jerk): {params['t_jerk']}")
    print(f"Acceleration Time (t_acc): {params['t_acc']}")
    print(f"Constant Velocity Time (t_constant_vel): {params['t_constant_vel']}")
    print(f"Maximum Velocity (max_vel): {params['max_vel']}")
    print(f"Maximum Acceleration (max_acc): {params['max_acc']}")
    print(f"Maximum Jerk (max_jerk): {params['max_jerk']}")
    print(f"Distance with Jerk (d_acc_jerk): {params['d_acc_jerk']}")
    print(f"Total Acceleration Distance (d_total_acc): {params['d_total_acc']}")
    print(f"Flat Distance (d_flat): {params['d_flat']}")
