def trapezoidal_motion_profile(start_pos, end_pos, max_vel, max_acc):
    """
    Generates a trapezoidal velocity profile analytically for a given motion.
    
    Args:
        start_pos (float): Initial position.
        end_pos (float): Final position.
        max_vel (float): Maximum velocity.
        max_acc (float): Maximum acceleration.
        
    Returns:
        dict: Contains arrays of time, position, velocity, and acceleration values.
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

    # Analytical expressions
    def position(t):
        if t < t_acc:  # Acceleration phase
            return start_pos + direction * 0.5 * max_acc * t**2
        elif t < t_acc + t_flat:  # Constant velocity phase
            return start_pos + direction * (d_acc + max_vel * (t - t_acc))
        else:  # Deceleration phase
            t_dec = t - t_acc - t_flat
            return start_pos + direction * (d_acc + max_vel * t_flat + max_vel * t_dec - 0.5 * max_acc * t_dec**2)

    def velocity(t):
        if t < t_acc:  # Acceleration phase
            return direction * max_acc * t
        elif t < t_acc + t_flat:  # Constant velocity phase
            return direction * max_vel
        else:  # Deceleration phase
            t_dec = t - t_acc - t_flat
            return direction * (max_vel - max_acc * t_dec)

    def acceleration(t):
        if t < t_acc:  # Acceleration phase
            return direction * max_acc
        elif t < t_acc + t_flat:  # Constant velocity phase
            return 0.0
        else:  # Deceleration phase
            return -direction * max_acc

    return {
        'total_time': total_time,
        'position': position,
        'velocity': velocity,
        'acceleration': acceleration
    }

# Example usage
if __name__ == '__main__':
    start_pos = 0
    end_pos = 10
    max_vel = 2
    max_acc = 1

    profile = trapezoidal_motion_profile(start_pos, end_pos, max_vel, max_acc)
    print("Total Time:", profile['total_time'])
    print("Position at t=1:", profile )
    print("Velocity at t=1:", profile )
    print("Acceleration at t=1:", profile )