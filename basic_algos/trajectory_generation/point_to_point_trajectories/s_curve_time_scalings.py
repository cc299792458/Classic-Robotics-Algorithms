# TODO: consider reach max velocity before reach max velocity
# TODO: consider short distance, so that the max velocity or max acceleration can't be reached

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
    
    # Time to reach max acceleration with given jerk
    t_jerk = max_acc / jerk    
    # Distance covered during one edge jerk phase
    d_jerk_edge = (1/6) * jerk * (t_jerk ** 3)
    # Velocity at the end of the jerk phase
    v_jerk = jerk * (t_jerk ** 2) / 2

    # Time to reach max velocity with maximum acceleration
    t_acc = (max_vel - 2 * v_jerk) / max_acc
    # Distance covered during one constant acceleration phase
    d_acc = v_jerk * t_acc + 0.5 * max_acc * t_acc**2
    # Velocity at the end of the acceleration phase
    v_acc = v_jerk + max_acc * t_acc

    # Distance covered during one middle jerk phase
    d_jerk_middle = v_acc * t_jerk + 0.5 * max_acc * t_jerk**2 - (1/6) * jerk * (t_jerk ** 3)

    # Total distance covered during acceleration and deceleration
    d_total_jerk = 2 * (d_jerk_edge + d_jerk_middle)
    d_total_acc = 2 * d_acc

    assert max_vel ** 2 > 4 * max_acc * jerk, "The system should reach max velocity after max acceleration."
    assert total_distance > d_total_jerk + d_total_acc, "The system should reach both max velocity and max acceleration in this distance."

    #Full S-curve Profile - Complete Motion Profile
    d_constant_vel = total_distance - d_total_jerk - d_total_acc
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
        'd_jerk_edge': d_jerk_edge,
        'd_jerk_middle': d_jerk_middle,
        'd_acc': d_acc,
        'd_constant_vel': d_constant_vel,
        'v_jerk': v_jerk,
        'v_acc': v_acc,
        'max_vel': max_vel,
        'max_acc': max_acc,
        'jerk': jerk,
        'start_pos': start_pos,
        'end_pos': end_pos
    }


if __name__ == '__main__':
    start_pos = 0
    end_pos = 10
    max_vel = 2
    max_acc = 1
    jerk = 0.5

    # Generate S-curve motion profile parameters
    params = s_curve_time_scalings(start_pos, end_pos, max_vel, max_acc, jerk)
    
    print("S-curve Motion Profile Parameters:")
    print(f"Total Time (t_total): {params['t_total']}")
    print(f"Jerk Time (t_jerk): {params['t_jerk']}")
    print(f"Acceleration Time (t_acc): {params['t_acc']}")
    print(f"Constant Velocity Time (t_constant_vel): {params['t_constant_vel']}")
    print(f"Maximum Velocity (max_vel): {params['max_vel']}")
    print(f"Maximum Acceleration (max_acc): {params['max_acc']}")
    print(f"Jerk: {params['jerk']}")
    print(f"Distance during Edge Jerk Phases (d_jerk_edge): {params['d_jerk_edge']}")
    print(f"Distance during Middle Jerk Phases (d_jerk_middle): {params['d_jerk_middle']}")
    print(f"Distance during Constant Acceleration Phases (d_acc): {params['d_acc']}")
    print(f"Constant Velocity Distance (d_constant_vel): {params['d_constant_vel']}")
    print(f"Velocity at the end of Jerk Phase (v_jerk): {params['v_jerk']}")
    print(f"Velocity at the end of Acceleration Phase (v_acc): {params['v_acc']}")

