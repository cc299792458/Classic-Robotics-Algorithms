import numpy as np

def third_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, duration=1.0):
    """
    Generates a third-order polynomial time scaling for a given start and end position.
    
    Args:
        start_pos (float): Initial position.
        end_pos (float): Final position.
        start_vel (float): Initial velocity.
        end_vel (float): Final velocity.
        duration (float): Time duration of the motion.
        
    Returns:
        A, B, C, D (tuple): Coefficients of the third-order polynomial.
    """
    # Define the system of equations for the third-order polynomial:
    # p(t) = A*t^3 + B*t^2 + C*t + D
    # v(t) = 3*A*t^2 + 2*B*t + C

    # Matrix of equations
    M = np.array([
        [0, 0, 0, 1],
        [duration**3, duration**2, duration, 1],
        [0, 0, 1, 0],
        [3 * duration**2, 2 * duration, 1, 0]
    ])

    # Results vector
    b = np.array([start_pos, end_pos, start_vel, end_vel])

    # Solve for coefficients
    coefficients = np.linalg.solve(M, b)

    return coefficients

def polynomial_via_point_trajectories(waypoints, velocities=None, duration=1.0):
    """
    Generates polynomial trajectories that pass through specified via points using third-order polynomials.
    
    Args:
        waypoints (list): List of positions the trajectory should pass through.
        velocities (list): List of velocities at the waypoints. Defaults to zeros at each waypoint.
        duration (float): Total time duration for the trajectory.
        
    Returns:
        list: List of coefficients for each segment of the trajectory.
    """
    if velocities is None:
        velocities = [0] * len(waypoints)

    assert len(waypoints) == len(velocities), "Waypoints and velocities must have the same length."
    
    num_segments = len(waypoints) - 1
    segment_duration = duration / num_segments
    trajectory_coefficients = []

    for i in range(num_segments):
        start_pos = waypoints[i]
        end_pos = waypoints[i + 1]
        start_vel = velocities[i]
        end_vel = velocities[i + 1]
        
        # Calculate the coefficients for the current segment
        coefficients = third_order_polynomial_time_scaling(
            start_pos, end_pos, start_vel, end_vel, duration=segment_duration
        )
        trajectory_coefficients.append(coefficients)
    
    return trajectory_coefficients

# Example usage
if __name__ == '__main__':
    waypoints = [0, 5, 10]
    velocities = [0, 0, 0]  # Zero velocities at all waypoints
    duration = 3
    coefficients_list = polynomial_via_point_trajectories(waypoints, velocities, duration)
    for idx, coeffs in enumerate(coefficients_list):
        print(f"Segment {idx} Coefficients (A, B, C, D):", coeffs)
