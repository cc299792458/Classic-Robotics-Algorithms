import numpy as np

from basic_algos.trajectory_generation.point_to_point_trajectories import third_order_polynomial_time_scaling, fifth_order_polynomial_time_scaling

# TODO: Add fifth order polynomial time scaling, too.

def polynomial_via_point_trajectories(waypoints, velocities=None, durations=None, order=3):
    """
    Generates polynomial trajectories for multi-dimensional waypoints.

    Args:
        waypoints (np.ndarray): Array of waypoints, shape (num_points, num_dimensions).
        velocities (np.ndarray, optional): Array of velocities at each waypoint, shape (num_points, num_dimensions).
        duration (float): Total duration of the trajectory.

    Returns:
        dict: Contains the trajectory coefficients for each dimension.
    """
    waypoints = np.asarray(waypoints)
    num_points, num_dimensions = waypoints.shape

    # If velocities are not provided, assume zero velocities at waypoints
    if velocities is None:
        velocities = np.zeros((num_points, num_dimensions))

    # Container for storing coefficients for each dimension
    coefficients = {dim: [] for dim in range(num_dimensions)}

    # Calculate coefficients for each dimension independently
    for dim in range(num_dimensions):
        # Extract waypoints and velocities for the current dimension
        wp = waypoints[:, dim]
        vel = velocities[:, dim]

        # Compute coefficients for each pair of waypoints
        for i in range(num_points - 1):
            start_pos = wp[i]
            end_pos = wp[i + 1]
            start_vel = vel[i]
            end_vel = vel[i + 1]
            duration = durations[i] if durations is not None else 1.0

            # Calculate the polynomial coefficients for this segment
            if order == 3:
                coeffs, _ = third_order_polynomial_time_scaling(start_pos, end_pos, start_vel, end_vel, duration=duration)
            elif order == 5:
                coeffs, _ = fifth_order_polynomial_time_scaling(start_pos, end_pos, start_vel, end_vel, duration=duration)
            else:
                raise NotImplementedError
            coefficients[dim].append(coeffs)

    return coefficients

if __name__ == '__main__':
    # Set up the example
    waypoints = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])  # Example for 2D
    velocities = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 0.0]])  # Initial and final velocities for each waypoint
    durations = np.array([1.0, 1.0, 1.0])

    # Generate coefficients for the polynomial trajectories
    coefficients = polynomial_via_point_trajectories(waypoints, velocities, durations)
    print(f"Coefficients: {coefficients}")
