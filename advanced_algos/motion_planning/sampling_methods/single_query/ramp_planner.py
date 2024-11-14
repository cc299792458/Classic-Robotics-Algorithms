import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from advanced_algos.motion_planning.smoothing import FSBAS

class RampPlanner:
    def __init__(self, start, goal, max_iters, collision_checker, position_limits, vmax, amax):
        """
        Initialize the RampPlanner.

        Args:
            start (np.ndarray): Initial state as a concatenated position and velocity vector.
                                 Shape: (2 * n_dimensions,)
            goal (np.ndarray): Goal state as a concatenated position and velocity vector.
                                Shape: (2 * n_dimensions,)
            max_iters (int): Maximum number of iterations for the planning loop.
            collision_checker (callable): Function to check for collisions.
                                          It should accept a state (np.ndarray) and return False if there is a collision,
                                          True otherwise.
            position_limits (tuple of np.ndarray): Position limits as (min_pos, max_pos).
                                                   Each is a NumPy array of shape (n_dimensions,).
            vmax (np.ndarray): Maximum velocity magnitudes for each dimension.
                                Shape: (n_dimensions,)
            amax (np.ndarray): Maximum acceleration magnitudes for each dimension.
                                Shape: (n_dimensions,)
        """
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.max_iters = max_iters
        self.collision_checker = collision_checker
        self.position_limits = position_limits  # (min_pos, max_pos)
        self.vmax = np.array(vmax, dtype=float)
        self.amax = np.array(amax, dtype=float)
        self.dimension = self.vmax.shape[0]

        # Initialize weights for weighted euclidean distance calculation
        self.init_weights()

        # Initialize forward and backward trees
        # Each tree consists of a list of states and a corresponding list of parent indices
        self.forward_tree = [self.start]
        self.forward_parents = [-1]  # -1 indicates no parent (root node)
        self.forward_segment_time = [None]
        self.forward_trajectory = [None]

        self.backward_tree = [self.goal]
        self.backward_parents = [-1]  # -1 indicates no parent (root node)
        self.backward_segment_time = [None]
        self.backward_trajectory = [None]

        # Store the final planned path
        self.path = None

        # Visualize or not
        self.visualization = False

    def init_weights(self):
        # Compute weights based on position and velocity limits
        min_pos, max_pos = self.position_limits
        position_weights = 1.0 / (max_pos - min_pos)**2  # Position weights
        velocity_weights = 1.0 / (self.vmax * 2)**2      # Velocity weights (assuming symmetrical limits)

        # Combine position and velocity weights
        self.weights = np.concatenate([position_weights, velocity_weights])

    def plan(self, visualize=False, visualization_args=None):
        """
        Plan the trajectory from the start to the goal state.

        Args:
            visualize (bool): Whether to visualize the planning process.
            visualization_args (dict or None): Additional arguments for visualization, such as obstacles.

        Returns:
            list of np.ndarray or None: Planned trajectory as a list of concatenated position and velocity vectors if successful, else None.
        """
        # Initialize visualization if needed
        if visualize:
            self.visualization = True
            self.visualization_args = visualization_args

            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self._initialize_plot()
    
        # Seed the trees with maximum braking trajectories
        self.seed_trees()

        # Main planning loop
        for _ in range(self.max_iters):
            # Extend the forward tree
            self.extend_tree(self.forward_tree, self.forward_parents, self.forward_segment_time, self.forward_trajectory)

            # Extend the backward tree
            self.extend_tree(self.backward_tree, self.backward_parents, self.backward_segment_time, self.backward_trajectory)

            # Check if the trees can be connected
            path = self.check_for_connection()
            if path:
                # Verify the trajectory is collision-free
                if self.check_collision(path):
                    # Apply path smoothing
                    smoothed_path = self.shortcut_path(path)
                    self.path = smoothed_path
                    return smoothed_path
                else:
                    # Remove infeasible edges from both trees
                    # Implementation depends on how infeasibility is detected
                    pass

        # Return None if no path is found within the maximum iterations
        return None
    
    def seed_trees(self):
        """
        Seed the forward and backward trees with maximum braking trajectories.
        """
        # Generate and add maximum braking trajectory to forward tree
        final_state, segment_time, trajectory_info = self.generate_max_braking_trajectory(self.start)
        if final_state is not None:
            nearby_nodes, _ = self._find_nearby_nodes(self.forward_tree, final_state, epsilon=1e-2)
            if nearby_nodes == []:
                self.forward_tree.append(final_state)
                self.forward_parents.append(len(self.forward_tree) - 2)  # Parent is the last node before adding
                if self.visualization:
                    self.forward_segment_time.append(segment_time)
                    self.forward_trajectory.append(trajectory_info)
                    self._update_plot()

        # Generate and add maximum reverse-braking trajectory to backward tree
        final_state, segment_time, trajectory_info = self.generate_max_braking_trajectory(self.goal)
        if final_state is not None:
            nearby_nodes, _ = self._find_nearby_nodes(self.backward_tree, final_state, epsilon=1e-2)
            if nearby_nodes == []:
                self.backward_tree.append(final_state)
                self.backward_parents.append(len(self.backward_tree) - 2)  # Parent is the last node before adding
                if self.visualization:
                    self.backward_segment_time.append(segment_time)
                    self.backward_trajectory.append(trajectory_info)
                    self._update_plot()

    def generate_max_braking_trajectory(self, state):
        """
        Generate the final state after applying maximum braking to reach zero velocity with synchronized stopping,
        considering each dimension independently.

        Args:
            state (np.ndarray): Initial state as a concatenated position and velocity vector.
                                Shape: (2 * n_dimensions,)

        Returns:
            tuple or None:
                - final_state (np.ndarray): Final state after braking.
                - trajectory_info (list of tuples): Trajectory information for each dimension:
                    - trajectory_type (str): Always "P+P-" for braking.
                    - acceleration (float): Absolute acceleration used for braking in the dimension.
        """
        pos_dim = self.dimension

        # Extract current position and velocity
        position = state[:pos_dim]
        velocity = state[pos_dim:]

        # Calculate stopping time for each dimension
        stopping_times = np.abs(velocity) / self.amax
        t_max = np.max(stopping_times)  # Use the maximum stopping time

        # Initialize arrays for final state and trajectory info
        final_position = np.zeros_like(position)
        final_velocity = np.zeros_like(velocity)
        trajectory_info = []

        # Calculate braking for each dimension independently
        for i in range(pos_dim):
            if t_max > 0:
                # Compute acceleration to stop in t_max
                a = -velocity[i] / t_max
                # Clamp acceleration to the allowable range
                a = np.clip(a, -self.amax[i], self.amax[i])

                # Compute position change using Δp = v * t + 0.5 * a * t^2
                delta_p = velocity[i] * t_max + 0.5 * a * t_max**2
                final_position[i] = position[i] + delta_p
                final_velocity[i] = 0.0  # Velocity reaches zero

                # Store trajectory information
                trajectory_info.append((np.abs(a), "P+P-"))
            else:
                # No motion in this dimension
                final_position[i] = position[i]
                final_velocity[i] = velocity[i]
                trajectory_info.append((0.0, "P+P-"))  # No acceleration needed

        # Combine final position and velocity into a single state
        final_state = np.concatenate((final_position, final_velocity))

        # Check position limits first
        if not self._check_state_limits(final_state):
            return None

        # Then check for collisions
        if not self.collision_checker(final_state):
            return None

        return final_state, t_max, trajectory_info

    def extend_tree(self, tree, parents, segment_times, trajectories):
        """
        Extend the given tree by sampling and connecting states, without checking connections.

        Args:
            tree (list of np.ndarray): The tree to extend, containing states as concatenated position and velocity vectors.
            parents (list of int): List of parent indices corresponding to each state in the tree.
            segment_times (List of float): List of time duration corresponding to each segment.
            trajectories (list of tuple): List of trajectory information regarding to each segment.

        Returns:
            None
        """
        # Step 1: Sample a state from the tree
        sampled_node = self._sample_tree_state(tree)

        # Step 2: Sample a random state from free space
        random_state = self._sample_free_space_state()

        # Step 3: Check collision for the sampled random state
        if not self.collision_checker(random_state):
            return  # Collision detected, skip this extension

        # Step 4: Add the new state to the tree
        tree.append(random_state)
        parents.append(tree.index(sampled_node))  # Record the parent index

        if self.visualization:
            segment_time, trajectory = self._compute_optimal_segment(sampled_node.reshape(2, 2), random_state.reshape(2, 2))
            segment_times.append(segment_time)
            trajectories.append(trajectory)
            self._update_plot()

    def check_for_connection(self):
        """
        Check if the forward and backward trees can be connected.

        Returns:
            list of np.ndarray or None: Combined trajectory if connection is found, else None.
        """
        # To be implemented
        return False

    def _combine_trajectories(self, f_index, b_index):
        """
        Combine forward and backward trajectories to form a complete path.

        Args:
            f_index (int): Index of the connecting node in the forward tree.
            b_index (int): Index of the connecting node in the backward tree.

        Returns:
            list of np.ndarray: Combined trajectory as a list of concatenated position and velocity vectors.
        """
        # To be implemented
        pass

    def remove_infeasible_edge(self, tree, child_index):
        """
        Remove the infeasible edge from the tree.

        Args:
            tree (list of np.ndarray): The tree from which to remove the edge.
            child_index (int): Index of the child node to remove.

        Returns:
            None
        """
        # To be implemented
        pass

    def _check_state_limits(self, state):
        """
        Check if the position part of the state is within the allowed limits.

        Args:
            state (np.ndarray): State as a concatenated position and velocity vector.
                                Shape: (2 * n_dimensions,)

        Returns:
            bool: True if within limits, False otherwise.
        """
        pos_indices = slice(0, self.dimension)  # Indices for position
        pos_min, pos_max = self.position_limits
        return np.all((state[pos_indices] >= pos_min) & (state[pos_indices] <= pos_max))
    
    def _weighted_euclidean_distance(self, state_from, state_to):
        """
        Calculate the weighted Euclidean distance between two states.

        Args:
            state_from (np.ndarray): The starting state as a concatenated position and velocity vector.
                                    Shape: (2 * n_dimensions,)
            state_to (np.ndarray): The target state as a concatenated position and velocity vector.
                                Shape: (2 * n_dimensions,)

        Returns:
            float: The weighted Euclidean distance between the two states.
        """
        diff = state_from - state_to  # Difference between the two states
        return np.sqrt(np.sum(self.weights * diff**2))  # Weighted Euclidean distance
    
    def _find_nearby_nodes(self, tree, target_node, epsilon):
        """
        Find all nodes in the tree that are within a given distance threshold using KDTree.

        Args:
            tree (list of np.ndarray): The tree containing existing nodes.
            target_node (np.ndarray): The target node to check against.
            epsilon (float): Distance threshold to consider nodes as "nearby".

        Returns:
            list of np.ndarray: List of nodes within the distance threshold.
            list of int: Indices of these nodes in the tree.
        """
        kdtree = KDTree(tree)
        nearby_indices = kdtree.query_ball_point(target_node, r=epsilon)
        nearby_nodes = [tree[i] for i in nearby_indices]
        return nearby_nodes, nearby_indices

    def _sample_tree_state(self, tree):
        """
        Randomly sample a state from the given tree.

        Args:
            tree (list of np.ndarray): The tree containing existing states.

        Returns:
            np.ndarray: A randomly sampled state from the tree.
        """
        return tree[np.random.randint(len(tree))]  # Uniformly sample a state from the tree
    
    def _sample_free_space_state(self):
        """
        Randomly sample a state from the free space with zero velocity.

        Returns:
            np.ndarray: A randomly sampled state with position in free space and zero velocity.
        """
        pos_dim = self.dimension
        # Sample position uniformly within position limits
        min_pos, max_pos = self.position_limits
        random_position = np.random.uniform(min_pos, max_pos)
        
        # Set velocity to zero
        zero_velocity = np.zeros(pos_dim)
        
        # Concatenate position and velocity
        return np.concatenate([random_position, zero_velocity])
    
    ############### Visualization ###############
    def _initialize_plot(self):
        """
        Initialize the plot with basic elements like start, goal, and obstacles.
        """
        # Set limits
        self.ax.set_xlim(self.position_limits[0][0], self.position_limits[1][0])
        self.ax.set_ylim(self.position_limits[0][1], self.position_limits[1][1])
        self.ax.set_title("Ramp Planner Visualization")
        self.ax.set_xlabel("Position X")
        self.ax.set_ylabel("Position Y")

        # Draw start and goal
        self.ax.scatter(self.start[0], self.start[1], c='green', label="Start")
        self._draw_velocity_arrow(self.start[:2], self.start[2:], 'green')  # Start velocity
        self.ax.scatter(self.goal[0], self.goal[1], c='red', label="Goal")
        self._draw_velocity_arrow(self.goal[:2], self.goal[2:], 'red')     # Goal velocity

        # Add obstacles if provided
        if self.visualization_args and "obstacles" in self.visualization_args:
            for obs in self.visualization_args["obstacles"]:
                self.ax.add_patch(plt.Rectangle(obs[:2], obs[2], obs[3], color="gray", alpha=0.5))

        self.ax.legend()

        plt.show(block=False)
        plt.pause(0.25)    

    def _draw_velocity_arrow(self, position, velocity, color):
        """
        Draw a velocity arrow at the given position.

        Args:
            ax (matplotlib.axes.Axes): The axis to draw on.
            position (np.ndarray): The position as [x, y].
            velocity (np.ndarray): The velocity as [vx, vy].
            color (str): The color of the arrow, matching the point color.
        """
        if velocity[0] == 0 and velocity[1] == 0:
            return
        # Scale arrow length for better visualization
        arrow_scale = 0.5  # Adjust as needed for visualization clarity
        self.ax.arrow(
            position[0], position[1],      # Starting point of the arrow
            velocity[0] * arrow_scale,     # Scaled x component of the velocity
            velocity[1] * arrow_scale,     # Scaled y component of the velocity
            head_width=0.15,                # Width of the arrowhead
            head_length=0.15,               # Length of the arrowhead
            fc=color, ec=color, alpha=0.2  # Face and edge color
        )

    def _update_plot(self):
        """
        Update the plot to visualize the current state of the trees.
        """
        # Clear the current axis
        self.ax.clear()

        # Redraw the basic elements
        self._initialize_plot()

        # Draw forward tree nodes and edges
        for i, (state, parent_index) in enumerate(zip(self.forward_tree, self.forward_parents)):
            if parent_index == -1:  # Skip the root node
                continue

            pos, vel = state[:2], state[2:]
            self.ax.plot(pos[0], pos[1], 'go', markersize=3, label="Forward Tree" if i == 1 else None)  # Nodes
            self._draw_velocity_arrow(position=pos, velocity=vel, color='green')

            # Draw edges
            parent_state = self.forward_tree[parent_index]
            segment_time = self.forward_segment_time[i]
            trajectory_info = self.forward_trajectory[i]
            self._draw_trajecotry_segment(parent_state, state, segment_time=segment_time, trajectory_info=trajectory_info, color='g-')
            
        # Draw backward tree nodes and edges
        for i, (state, parent_index) in enumerate(zip(self.backward_tree, self.backward_parents)):
            if parent_index == -1:  # Skip the root node
                continue

            pos, vel = state[:2], state[2:]
            self.ax.plot(pos[0], pos[1], 'ro', markersize=3, label="Backward Tree" if i == 1 else None)  # Nodes
            self._draw_velocity_arrow(position=pos, velocity=vel, color='red')

            # Draw edges
            parent_state = self.backward_tree[parent_index]
            segment_time = self.backward_segment_time[i]
            trajectory_info = self.backward_trajectory[i]
            self._draw_trajecotry_segment(parent_state, state, segment_time=segment_time, trajectory_info=trajectory_info, color='r-')

        # Add legend if not too cluttered
        handles, labels = self.ax.get_legend_handles_labels()
        if len(labels) <= 10:
            self.ax.legend(loc="upper right")

        plt.show(block=False)
        plt.pause(0.25)

    def _draw_trajecotry_segment(self, state_from, state_to, segment_time, trajectory_info, color):
        positions, velocities = [], []
        times = np.linspace(0, np.sum(segment_time), 10)
        for t in times:
            state = self._get_state_in_segment(
                start_state=state_from.reshape(2, 2), end_state=state_to.reshape(2, 2), 
                segment_time=segment_time, segment_trajectory=trajectory_info, t=t
            )
            positions.append(state[0])
            velocities.append(state[1])

        positions = np.array(positions)
        self.ax.plot(positions[:, 0], positions[:, 1], color, linewidth=0.5)

    ############### Trajectory Generation ###############
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
        position = np.zeros(self.dimension)
        velocity = np.zeros(self.dimension)

        for dim in range(self.dimension):
            acc, trajectory_type = segment_trajectory[dim]
            x1, x2 = start_state[0][dim], end_state[0][dim]
            v1, v2 = start_state[1][dim], end_state[1][dim]

            pos, vel = self._compute_trajectory_state(
                x1=x1, x2=x2, v1=v1, v2=v2, a=acc,
                vmax=self.vmax[dim], T=segment_time,
                trajectory_type=trajectory_type, t=t
            )
            position[dim] = pos
            velocity[dim] = vel

        return position, velocity

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

    def _compute_optimal_segment(self, start_state, end_state):
        segment_time = self._calculate_segment_time(start_state, end_state)
        segment_trajectory = self._calculate_segment_trajectory(
                start_state, end_state, segment_time
            )
        
        return segment_time, segment_trajectory
    
    def _calculate_segment_time(self, start_state, end_state, safe_margin=1e-6):
        """
        Calculate the maximum time required to traverse a segment across all dimensions,
        considering vmax and amax constraints.
        """
        t_requireds = np.array([
            self._univariate_time_optimal_interpolants(
                start_pos=start_state[0][dim],
                end_pos=end_state[0][dim],
                start_vel=start_state[1][dim],
                end_vel=end_state[1][dim],
                vmax=self.vmax[dim],
                amax=self.amax[dim]
            )
            for dim in range(self.dimension)
        ])
        return np.max(t_requireds) + safe_margin

    def _calculate_segment_trajectory(self, start_state, end_state, segment_time):
        """
        Calculate the trajectory for a single segment using minimum acceleration interpolants.
        """
        # Vectorized calculation for all dimensions
        trajectory_data = [
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
        return np.array(trajectory_data, dtype=object)

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
