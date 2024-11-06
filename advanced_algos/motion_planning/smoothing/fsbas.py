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
            
            t_acc_to_vmax = abs(self.vmax[dim] - v_start) / self.amax[dim]
            dist_acc_to_vmax = v_start * t_acc_to_vmax + 0.5 * self.amax[dim] * t_acc_to_vmax**2

            t_dec_from_vmax = abs(self.vmax[dim] - v_end) / self.amax[dim]
            dist_dec_from_vmax = v_end * t_dec_from_vmax + 0.5 * self.amax[dim] * t_dec_from_vmax**2

            if abs(delta_pos) <= dist_acc_to_vmax + dist_dec_from_vmax:
                t_acc = (-v_start + np.sqrt(v_start**2 + 2 * self.amax[dim] * abs(delta_pos) / 2)) / self.amax[dim]
                t_required = t_acc + (v_end - v_start) / self.amax[dim]
            else:
                dist_remaining = abs(delta_pos) - (dist_acc_to_vmax + dist_dec_from_vmax)
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
