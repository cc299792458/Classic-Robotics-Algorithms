"""
This implementation is based on the method presented in the paper:
"Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts"
"""

import numpy as np

from scipy.interpolate import CubicSpline

class FSBAS:
    """
    Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts.
    This class implements a smoothing algorithm for manipulator trajectories with bounded velocity 
    and acceleration, using optimal shortcuts for improved performance and natural-looking motion.
    """
    
    def __init__(self, path, vmax, amax, collision_checker, max_iterations=100):
        self.path = path
        self.vmax = np.array(vmax)
        self.amax = np.array(amax)
        self.collision_checker = collision_checker
        self.max_iterations = max_iterations
        self.segment_trajectory = []  # Store the optimal trajectory functions for each segment
        self.segment_time = []  # Store the time duration for each segment

    def smooth_path(self):
        """
        Perform smoothing by first converting milestones into a time-optimal trajectory and 
        then applying shortcutting. 

        Return:
        - A smoothed path as a list of waypoints.
        """
        # Generate initial smoothed trajectory using time-optimal segments
        self._generate_initial_time_optimal_trajectory()
        total_time = sum(self.segment_time)

        for _ in range(self.max_iterations):
            t1, t2 = self._select_random_times(total_time)
            if t1 >= t2:
                continue

            start_state = self._get_state_at_time(t1)
            end_state = self._get_state_at_time(t2)
            shortcut = self._compute_optimal_shortcut(start_state, end_state)

            if shortcut and self._is_collision_free(shortcut):
                self._update_segment_data(t1, t2, shortcut)

        return self.path

    def _select_random_times(self, total_time):
        """
        Select two random times within the total trajectory duration.

        Input:
        - total_time: The total duration of the trajectory

        Return:
        - t1, t2: Two random times within the total trajectory duration.
        """
        t1 = np.random.uniform(0, total_time)
        t2 = np.random.uniform(0, total_time)
        return min(t1, t2), max(t1, t2)

    def _get_state_at_time(self, t):
        """
        Find the state at a specific time t within the trajectory.

        Input:
        - t: The target time

        Return:
        - The interpolated state (position, velocity) at time t.
        """
        elapsed_time = 0
        for i, segment_duration in enumerate(self.segment_time):
            if elapsed_time + segment_duration >= t:
                relative_t = t - elapsed_time
                position_func, velocity_func = self.segment_trajectory[i]
                return (position_func(relative_t), velocity_func(relative_t))
            elapsed_time += segment_duration
        return None

    def _update_segment_data(self, t1, t2, shortcut):
        """
        Update the segment trajectory and time information after applying a shortcut.

        Input:
        - t1, t2: Start and end times of the shortcut in the original trajectory
        - shortcut: The new trajectory segment to replace the original one
        """
        # Clear and regenerate path based on shortcut (simplified update)
        self.path = [self._get_state_at_time(t) for t in np.linspace(t1, t2, num=len(shortcut))]

        # Recompute segment trajectories and times based on updated path
        self.segment_trajectory.clear()
        self.segment_time.clear()
        self._generate_initial_time_optimal_trajectory()

    def _calculate_segment_time(self, start_state, end_state):
        times = []
        for dim in range(len(start_state[0])):
            delta_pos = end_state[0][dim] - start_state[0][dim]
            v_start, v_end = start_state[1][dim], end_state[1][dim]
            
            t_acc_to_vmax = np.abs(self.vmax[dim] - v_start) / self.amax[dim]
            dist_acc_to_vmax = v_start * t_acc_to_vmax + 0.5 * self.amax[dim] * t_acc_to_vmax**2

            t_dec_from_vmax = np.abs(self.vmax[dim] - v_end) / self.amax[dim]
            dist_dec_from_vmax = v_end * t_dec_from_vmax + 0.5 * self.amax[dim] * t_dec_from_vmax**2

            if np.abs(delta_pos) <= dist_acc_to_vmax + dist_dec_from_vmax:
                t_acc = (-v_start + np.sqrt(v_start**2 + 2 * self.amax[dim] * np.abs(delta_pos) / 2)) / self.amax[dim]
                t_required = t_acc + (v_end - v_start) / self.amax[dim]
            else:
                dist_remaining = np.abs(delta_pos) - (dist_acc_to_vmax + dist_dec_from_vmax)
                t_const = dist_remaining / self.vmax[dim]
                t_required = t_acc_to_vmax + t_const + t_dec_from_vmax

            times.append(t_required)

        return max(times)

    def _generate_cubic_spline_segment(self, start_state, end_state, segment_time):
        position_funcs, velocity_funcs = [], []

        for dim in range(len(start_state[0])):
            spline = CubicSpline(
                [0, segment_time], 
                [start_state[0][dim], end_state[0][dim]], 
                bc_type=((1, start_state[1][dim]), (1, end_state[1][dim]))
            )
            position_funcs.append(spline)
            velocity_funcs.append(spline.derivative())

        def position_func(t):
            return np.array([f(t) for f in position_funcs])

        def velocity_func(t):
            return np.array([f(t) for f in velocity_funcs])

        return position_func, velocity_func

    def _generate_initial_time_optimal_trajectory(self):
        self.segment_trajectory = []
        self.segment_time = []

        for i in range(len(self.path) - 1):
            start_state, end_state = self.path[i], self.path[i + 1]
            segment_time = self._calculate_segment_time(start_state, end_state)
            position_func, velocity_func = self._generate_cubic_spline_segment(start_state, end_state, segment_time)
            self.segment_trajectory.append((position_func, velocity_func))
            self.segment_time.append(segment_time)

    def _is_collision_free(self, trajectory_segment):
        for state in trajectory_segment:
            if not self.collision_checker(state):
                return False
        return True
    
    def _univariate_time_optimal_interpolants(self, x1, x2, v1, v2, vmax, amax):
        """
        Compute the time-optimal trajectory execution time for univariate motion.

        Input:
        - x1, x2: Initial and final positions.
        - v1, v2: Initial and final velocities.
        - vmax: Maximum velocity.
        - amax: Maximum acceleration.

        Return:
        - T: Minimal execution time for valid motion primitive combinations, or None if no valid combination exists.
        """
        def solve_quadratic(a, b, c):
            """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return []
            sqrt_discriminant = np.sqrt(discriminant)
            return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]
        
        # Class P^+P^-
        def compute_p_plus_p_minus():
            coefficients = [
                amax,
                2 * v1,
                (v1**2 - v2**2) / (2 * amax) + x1 - x2
            ]
            solutions = solve_quadratic(*coefficients)
            valid_t = [t for t in solutions if max((v2 - v1) / amax, 0) <= t <= (vmax - v1) / amax]
            if not valid_t:
                return None
            t_p = valid_t[0]
            
            return np.array(2 * t_p + (v1 - v2) / amax)

        # Class P^-P^+
        def compute_p_minus_p_plus():
            coefficients = [
                amax,
                -2 * v1,
                (v1**2 - v2**2) / (2 * amax) + x2 - x1
            ]
            solutions = solve_quadratic(*coefficients)
            valid_t = [t for t in solutions if max((v1 - v2) / amax, 0) <= t <= (vmax + v1) / amax]
            if not valid_t:
                return None
            t_p = valid_t[0]
            
            return np.array(2 * t_p + (v2 - v1) / amax)

        # Class P^+L^+P^-
        def compute_p_plus_l_plus_p_minus():
            t_p1 = (vmax - v1) / amax
            t_p2 = (vmax - v2) / amax
            t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) + (x2 - x1) / vmax
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return np.array(t_p1 + t_l + t_p2)

        # Class P^-L^+P^+
        def compute_p_minus_l_plus_p_plus():
            t_p1 = (vmax + v1) / amax
            t_p2 = (vmax + v2) / amax
            t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) - (x2 - x1) / vmax
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return np.array(t_p1 + t_l + t_p2)

        # Evaluate all four classes in the specified order
        t_p_plus_p_minus = compute_p_plus_p_minus()
        t_p_minus_p_plus = compute_p_minus_p_plus()
        t_p_plus_l_plus_p_minus = compute_p_plus_l_plus_p_minus()
        t_p_minus_l_plus_p_plus = compute_p_minus_l_plus_p_plus()

        # Collect valid times and return the minimal one
        times = [t for t in [t_p_plus_p_minus, t_p_minus_p_plus, t_p_plus_l_plus_p_minus, t_p_minus_l_plus_p_plus] if t is not None]
        return np.min(times) if times else None
    
    def _minimum_acceleration_interpolants(x1, x2, v1, v2, vmax, T):
        """
        Compute the minimum-acceleration trajectory for fixed end time T.

        Input:
        - x1, x2: Initial and final positions.
        - v1, v2: Initial and final velocities.
        - vmax: Maximum velocity.
        - T: Fixed end time.

        Return:
        - a_min: Minimal acceleration for valid motion primitive combinations, or None if no valid combination exists.
        """
        def solve_quadratic(a, b, c):
            """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return []
            sqrt_discriminant = np.sqrt(discriminant)
            return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]

        # Class P^+P^-
        def compute_p_plus_p_minus():
            coefficients = [
                T**2,
                2 * T * (v1 + v2) + 4 * (x1 - x2),
                -(v2 - v1)**2
            ]
            solutions = solve_quadratic(*coefficients)
            valid_a = []
            for a in solutions:
                if a <= 0:
                    continue
                t_s = 0.5 * (T + (v2 - v1) / a)
                if 0 < t_s < T and abs(v1 + a * t_s) <= vmax:
                    valid_a.append(a)
            return min(valid_a) if valid_a else None

        # Class P^-P^+
        def compute_p_minus_p_plus():
            coefficients = [
                T**2,
                -2 * T * (v1 + v2) - 4 * (x1 - x2),
                -(v2 - v1)**2
            ]
            solutions = solve_quadratic(*coefficients)
            valid_a = []
            for a in solutions:
                if a <= 0:
                    continue
                t_s = 0.5 * (T + (v1 - v2) / a)
                if 0 < t_s < T and abs(v1 - a * t_s) <= vmax:
                    valid_a.append(a)
            return min(valid_a) if valid_a else None

        # Class P^+L^+P^-
        def compute_p_plus_l_plus_p_minus():
            a = (vmax**2 - vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax - (x2 - x1))
            if a <= 0:
                return None
            t_p1 = (vmax - v1) / a
            t_p2 = (vmax - v2) / a
            t_l = T - t_p1 - t_p2
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return a

        # Class P^-L^-P^+
        def compute_p_minus_l_minus_p_plus():
            a = (vmax**2 + vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax + (x2 - x1))
            if a <= 0:
                return None
            t_p1 = (vmax + v1) / a
            t_p2 = (vmax + v2) / a
            t_l = T - t_p1 - t_p2
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return a

        # Evaluate all four classes independently
        results = [
            compute_p_plus_p_minus(),  # P^+P^-
            compute_p_minus_p_plus(),  # P^-P^+
            compute_p_plus_l_plus_p_minus(),  # P^+L^+P^-
            compute_p_minus_l_minus_p_plus()  # P^-L^-P^+
        ]
        valid_results = [a for a in results if a is not None]

        return min(valid_results) if valid_results else None

if __name__ == "__main__":
    """
    Demo for the FSBAS class:
    - Create a simple path with two dimensions.
    - Define maximum velocity and acceleration.
    - Perform trajectory smoothing and output results.
    """
    def collision_checker(state):
        # Dummy collision checker: always returns True (no collisions)
        return True

    # Example path: [(position, velocity)] for each waypoint
    path = [
        ([0.0, 0.0], [0.0, 0.0]),
        ([1.0, 2.0], [0.0, 0.0]),
        ([3.0, 3.0], [0.0, 0.0]),
        ([4.0, 0.0], [0.0, 0.0])
    ]

    # Maximum velocity and acceleration for each dimension
    vmax = [2.0, 2.0]  # [vmax_x, vmax_y]
    amax = [1.0, 1.0]  # [amax_x, amax_y]

    # Initialize the FSBAS class
    fsbas = FSBAS(path, vmax, amax, collision_checker, max_iterations=10)

    # Perform smoothing
    smoothed_path = fsbas.smooth_path()

    # Print the smoothed path
    print("\nSmoothed Path:")
    for state in smoothed_path:
        print(f"Position: {state[0]}, Velocity: {state[1]}")

    # Print the trajectory segment times
    print("\nSegment Times:")
    print(fsbas.segment_time)

    # Print an example state at a given time
    example_time = sum(fsbas.segment_time) / 2  # Midpoint of the total trajectory
    state_at_example_time = fsbas._get_state_at_time(example_time)
    print(f"\nState at time {example_time:.2f}:")
    print(f"Position: {state_at_example_time[0]}, Velocity: {state_at_example_time[1]}")
