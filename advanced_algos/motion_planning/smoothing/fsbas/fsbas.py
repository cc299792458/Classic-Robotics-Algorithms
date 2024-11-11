"""
This implementation is based on the method presented in the paper:
"Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts"
"""

import numpy as np

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
        self.dimension = self.vmax.shape[0]
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
        self._generate_initial_trajectory()
        total_time = sum(self.segment_time)

        for _ in range(self.max_iterations):
            t1, t2 = self._select_random_times(total_time)

            start_state = self._get_state_at_time(t1)
            end_state = self._get_state_at_time(t2)
            shortcut_time, shortcut_trajectory = self._compute_optimal_shortcut(start_state, end_state)

            if shortcut_trajectory and self._is_segment_collision_free(start_state, end_state, shortcut_time, shortcut_trajectory):
                self._update_segment_data(start_state, end_state, t1, t2, shortcut_time, shortcut_trajectory)

        return self.path
    
    def _generate_initial_trajectory(self):
        """Generate the initial time-optimal trajectory for all segments."""
        self.segment_trajectory = []
        self.segment_time = []

        for i in range(len(self.path) - 1):
            start_state, end_state = self.path[i], self.path[i + 1]
            segment_time = self._calculate_segment_time(start_state, end_state)
            self.segment_time.append(segment_time)

            segment_trajectory = self._calculate_segment_trajectory(
                start_state, end_state, segment_time
            )
            self.segment_trajectory.append(segment_trajectory)

    def _calculate_segment_time(self, start_state, end_state):
        """
        Calculate the time required to traverse a segment between start_state and end_state
        for each dimension, considering vmax and amax constraints.
        """
        t_requireds = [
            self._univariate_time_optimal_interpolants(
                start_pos=start_state[0][dim],
                end_pos=end_state[0][dim],
                start_vel=start_state[1][dim],
                end_vel=end_state[1][dim],
                vmax=self.vmax[dim],
                amax=self.amax[dim]
            )
            for dim in range(self.dimension)
        ]
        return max(t_requireds)

    def _calculate_segment_trajectory(self, start_state, end_state, segment_time):
        """
        Calculate the trajectory for a single segment using minimum acceleration interpolants.
        """
        return [
            self._minimum_acceleration_interpolants(
                start_pos=start_state[0][dim],
                end_pos=end_state[0][dim],
                start_vel=start_state[1][dim],
                end_vel=end_state[1][dim],
                vmax=self.vmax[dim],
                T=segment_time
            )
            for dim in range(self.dimension)
        ]
    
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
        Find the interpolated state (position, velocity) at a specific time t within the trajectory.

        Input:
        - t: The target time

        Return:
        - position: Numpy array of positions at time t.
        - velocity: Numpy array of velocities at time t.
        """
        elapsed_time = 0

        for i, (segment_time, segment_trajectory) in enumerate(zip(self.segment_time, self.segment_trajectory)):
            if elapsed_time + segment_time >= t:
                # Relative time within the segment
                relative_t = t - elapsed_time
                start_state = self.path[i]
                end_state = self.path[i + 1]

                # Compute position and velocity
                return self._get_state_in_segment(start_state, end_state, segment_time, segment_trajectory, relative_t)

            elapsed_time += segment_time

        # Return None if time t is beyond the total trajectory duration
        return None

    def _get_state_in_segment(self, start_state, end_state, segment_time, segment_trajectory, t):
        """
        Compute the state (position, velocity) within a segment at time t.

        Input:
        - start_state: The starting state of the segment.
        - end_state: The ending state of the segment.
        - segment_time: The duration of the segment.
        - segment_trajectory: Trajectory parameters for each dimension.
        - t: Relative time within the segment.

        Return:
        - position: Numpy array of positions at time t.
        - velocity: Numpy array of velocities at time t.
        """
        position, velocity = zip(*[
            self._compute_trajectory_state(
                x1=start_state[0][dim],
                x2=end_state[0][dim],
                v1=start_state[1][dim],
                v2=end_state[1][dim],
                a=acc,
                vmax=self.vmax[dim],
                T=segment_time,
                trajectory_type=trajectory_type,
                t=t
            )
            for dim, (acc, trajectory_type) in enumerate(segment_trajectory)
        ])

        return np.array(position), np.array(velocity)
    
    def _compute_optimal_shortcut(self, start_state, end_state):
        segment_time = self._calculate_segment_time(start_state, end_state)
        segment_trajectory = self._calculate_segment_trajectory(
                start_state, end_state, segment_time
            )
        
        return segment_time, segment_trajectory


    def _compute_trajectory_state(self, x1, x2, v1, v2, a, vmax, T, trajectory_type, t):
        """
        Compute the position x(t) and velocity v(t) for a given trajectory type.

        Input:
        - x1, x2: Initial and final positions.
        - v1, v2: Initial and final velocities.
        - a: Acceleration used in the trajectory.
        - vmax: Maximum velocity.
        - T: Total trajectory time.
        - trajectory_type: One of 'P+P-', 'P-P+', 'P+L+P-', 'P-L-P+'.
        - t: The time at which to compute the state.

        Return:
        - x_t: Position at time t.
        - v_t: Velocity at time t.
        """
        if trajectory_type == 'P+P-':
            # Compute switch time
            t_s = 0.5 * (T + (v2 - v1) / a)
            if t <= t_s:  # Acceleration phase
                v_t = v1 + a * t
                x_t = x1 + v1 * t + 0.5 * a * t**2
            else:  # Deceleration phase
                delta_t = t - t_s
                v_peak = v1 + a * t_s
                v_t = v_peak - a * delta_t
                x_t = (x1 + v1 * t_s + 0.5 * a * t_s**2 +
                    v_peak * delta_t - 0.5 * a * delta_t**2)

        elif trajectory_type == 'P-P+':
            # Compute switch time
            t_s = 0.5 * (T + (v1 - v2) / a)
            if t <= t_s:  # Deceleration phase
                v_t = v1 - a * t
                x_t = x1 + v1 * t - 0.5 * a * t**2
            else:  # Acceleration phase
                delta_t = t - t_s
                v_valley = v1 - a * t_s
                v_t = v_valley + a * delta_t
                x_t = (x1 + v1 * t_s - 0.5 * a * t_s**2 +
                    v_valley * delta_t + 0.5 * a * delta_t**2)

        elif trajectory_type == 'P+L+P-':
            # Compute durations
            t_p1 = (vmax - v1) / a
            t_p2 = (vmax - v2) / a
            t_l = T - t_p1 - t_p2
            if t <= t_p1:  # Acceleration phase
                v_t = v1 + a * t
                x_t = x1 + v1 * t + 0.5 * a * t**2
            elif t <= t_p1 + t_l:  # Constant velocity phase
                delta_t = t - t_p1
                v_t = vmax
                x_t = (x1 + v1 * t_p1 + 0.5 * a * t_p1**2 +
                    vmax * delta_t)
            else:  # Deceleration phase
                delta_t = t - t_p1 - t_l
                v_t = vmax - a * delta_t
                x_t = (x1 + v1 * t_p1 + 0.5 * a * t_p1**2 +
                    vmax * t_l +
                    vmax * delta_t - 0.5 * a * delta_t**2)

        elif trajectory_type == 'P-L-P+':
            # Compute durations
            t_p1 = (vmax + v1) / a
            t_p2 = (vmax + v2) / a
            t_l = T - t_p1 - t_p2
            if t <= t_p1:  # Deceleration phase
                v_t = v1 - a * t
                x_t = x1 + v1 * t - 0.5 * a * t**2
            elif t <= t_p1 + t_l:  # Constant negative velocity phase
                delta_t = t - t_p1
                v_t = -vmax
                x_t = (x1 + v1 * t_p1 - 0.5 * a * t_p1**2 +
                    (-vmax) * delta_t)
            else:  # Acceleration phase
                delta_t = t - t_p1 - t_l
                v_t = -vmax + a * delta_t
                x_t = (x1 + v1 * t_p1 - 0.5 * a * t_p1**2 +
                    (-vmax) * t_l +
                    (-vmax) * delta_t + 0.5 * a * delta_t**2)

        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        return x_t, v_t

    def _update_segment_data(self, t1, t2, shortcut_time, shortcut_trajectory):
        """
        Update the trajectory u by replacing the section from t1 to t2 with the shortcut.

        Input:
        - t1, t2: Start and end times of the shortcut in the original trajectory.
        - shortcut_time: Duration of the shortcut segment.
        - shortcut_trajectory: Trajectory of the shortcut segment.
        """
        # Step 1: Find the indices of segments affected by t1 and t2
        elapsed_time = 0
        start_index, end_index = None, None
        for i, segment_time in enumerate(self.segment_time):
            if elapsed_time <= t1 < elapsed_time + segment_time:
                start_index = i
            if elapsed_time <= t2 <= elapsed_time + segment_time:
                end_index = i
            elapsed_time += segment_time

        # Step 2: Split the original trajectory into three parts
        # - Part before t1
        before_trajectory = self.segment_trajectory[:start_index]
        before_time = self.segment_time[:start_index]
        # - Part after t2
        after_trajectory = self.segment_trajectory[end_index + 1:]
        after_time = self.segment_time[end_index + 1:]

        # Step 3: Insert the shortcut into the trajectory
        new_trajectory = before_trajectory + [shortcut_trajectory] + after_trajectory
        new_time = before_time + [shortcut_time] + after_time

        # Step 4: Update the class attributes
        self.segment_trajectory = new_trajectory
        self.segment_time = new_time

        # Update the path to reflect the new shortcut endpoints
        self.path = (
            self.path[:start_index + 1] +  # Keep unaffected path before t1
            [self._get_state_at_time(t1)[0], self._get_state_at_time(t2)[0]] +  # Insert shortcut endpoints
            self.path[end_index + 1:]     # Keep unaffected path after t2
        )
    
    def _univariate_time_optimal_interpolants(self, start_pos, end_pos, start_vel, end_vel, vmax, amax):
        """
        Compute the time-optimal trajectory execution time for univariate motion.

        Input:
        - start_pos, end_pos: Initial and final positions.
        - start_vel, end_vel: Initial and final velocities.
        - vmax: Maximum velocity.
        - amax: Maximum acceleration.

        Return:
        - T: Minimal execution time for valid motion primitive combinations, or None if no valid combination exists.
        """
        x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel

        def solve_quadratic(a, b, c):
            """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return []
            sqrt_discriminant = np.sqrt(discriminant)
            return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]
        
        # Class P+P-
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

        # Class P-P+
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

        # Class P+L+P-
        def compute_p_plus_l_plus_p_minus():
            t_p1 = (vmax - v1) / amax
            t_p2 = (vmax - v2) / amax
            t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) + (x2 - x1) / vmax
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return np.array(t_p1 + t_l + t_p2)

        # Class P-L+P+
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
    
    def _minimum_acceleration_interpolants(self, start_pos, end_pos, start_vel, end_vel, vmax, T):
        """
        Compute the minimum-acceleration trajectory for fixed end time T.

        Input:
        - start_pos, end_pos: Initial and final positions.
        - start_vel, end_vel: Initial and final velocities.
        - vmax: Maximum velocity.
        - T: Fixed end time.

        Return:
        - a_min: Minimal acceleration for valid motion primitive combinations, or None if no valid combination exists.
        - selected_primitive: Name of the selected motion primitive.
        """
        x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel

        def solve_quadratic(a, b, c):
            """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return []
            sqrt_discriminant = np.sqrt(discriminant)
            return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]

        # Class P+P-
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
            return (min(valid_a), 'P+P-') if valid_a else None

        # Class P-P+
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
            return (min(valid_a), 'P-P+') if valid_a else None

        # Class P+L+P-
        def compute_p_plus_l_plus_p_minus():
            a = (vmax**2 - vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax - (x2 - x1))
            if a <= 0:
                return None
            t_p1 = (vmax - v1) / a
            t_p2 = (vmax - v2) / a
            t_l = T - t_p1 - t_p2
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return (a, 'P+L+P-')

        # Class P-L-P+
        def compute_p_minus_l_minus_p_plus():
            a = (vmax**2 + vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax + (x2 - x1))
            if a <= 0:
                return None
            t_p1 = (vmax + v1) / a
            t_p2 = (vmax + v2) / a
            t_l = T - t_p1 - t_p2
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return (a, 'P-L-P+')

        # Evaluate all four classes independently
        results = [
            compute_p_plus_p_minus(),  # P+P-
            compute_p_minus_p_plus(),  # P-P+
            compute_p_plus_l_plus_p_minus(),  # P+L+P-
            compute_p_minus_l_minus_p_plus()  # P-L-P+
        ]
        valid_results = [result for result in results if result is not None]

        if not valid_results:
            return None, None

        # Find the minimum acceleration and corresponding primitive
        a_min, selected_primitive = min(valid_results, key=lambda x: x[0])
        return a_min, selected_primitive

    def _is_segment_collision_free(self, start_state, end_state, segment_time, segment_trajectory):
        sampled_times = np.linspace(0, segment_time, 10)

        for time in sampled_times:
            state = self._get_state_in_segment(
                start_state=start_state, end_state=end_state, 
                segment_time=segment_time, segment_trajectory=segment_trajectory,
                t=time
            )
            if not self.collision_checker(state):
                return False
            return True

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
