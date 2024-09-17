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
    t_acc = max_acc / max_jerk

    # Distance and velocity when reaching max acceleration
    d_acc_jerk = 0.5 * max_jerk * t_acc ** 2
    v_acc_jerk = max_jerk * t_acc ** 2

    # Time to reach max velocity with maximum acceleration
    t_vel = (max_vel - v_acc_jerk) / max_acc
    d_vel_acc = v_acc_jerk * t_vel + 0.5 * max_acc * t_vel ** 2

    # Total distance to reach maximum velocity
    d_total_acc = 2 * (d_acc_jerk + d_vel_acc)

    if total_distance < d_total_acc:
        # Case 1: Triangular S-curve Profile
        # Cannot reach max velocity and max acceleration, compute a shorter profile
        # Adjust max_acc and max_vel to fit within the total distance
        max_acc = (max_jerk * total_distance) ** (1/3)
        t_acc = max_acc / max_jerk
        max_vel = max_jerk * t_acc ** 2
        t_vel = 0  # No constant velocity phase

        # Update distances
        d_acc_jerk = 0.5 * max_jerk * t_acc ** 2
        d_total_acc = 2 * d_acc_jerk

    else:
        # Case 2: Full S-curve Profile
        # Can reach max velocity, compute the distances and times
        if total_distance < (d_total_acc + max_vel * t_vel):
            # Case 3: Modified S-curve Profile
            # Cannot reach maximum velocity but can reach maximum acceleration
            max_vel = (total_distance - d_total_acc) / t_vel

        d_constant_vel = total_distance - d_total_acc
        t_constant_vel = d_constant_vel / max_vel

    # Compute the times for different phases
    t_total_acc = 2 * t_acc + t_vel
    t_total = 2 * t_total_acc + t_constant_vel

    return {
        't_acc': t_acc,
        't_vel': t_vel,
        't_total': t_total,
        'd_acc_jerk': d_acc_jerk,
        'd_vel_acc': d_vel_acc,
        'd_total_acc': d_total_acc,
        'd_constant_vel': d_constant_vel if 'd_constant_vel' in locals() else 0
    }

# Example usage
if __name__ == '__main__':
    params = s_curve_time_scalings(total_distance=10, max_vel=2, max_acc=1, max_jerk=0.5)
    print(params)