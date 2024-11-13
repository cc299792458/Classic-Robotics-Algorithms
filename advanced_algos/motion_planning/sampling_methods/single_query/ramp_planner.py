import numpy as np
import random

class RampPlanner:
    def __init__(self, start, goal, max_iters, collision_checker, state_limits, control_limits):
        """
        Initialize the RampPlanner.

        Args:
            start (tuple of np.ndarray): Initial state as (position, velocity).
            goal (tuple of np.ndarray): Goal state as (position, velocity).
            max_iters (int): Maximum number of iterations for the planning loop.
            collision_checker (callable): Function or object to check for collisions.
                                          It should accept a state and return True if there is a collision.
            state_limits (dict): State limits, e.g.,
                                 {
                                     'position': (np.array([min_pos1, min_pos2, ...]), 
                                                  np.array([max_pos1, max_pos2, ...])),
                                     'velocity': (np.array([min_vel1, min_vel2, ...]), 
                                                  np.array([max_vel1, max_vel2, ...]))
                                 }
            control_limits (dict): Control input limits, e.g.,
                                   {
                                       'acceleration': (np.array([min_acc1, min_acc2, ...]), 
                                                       np.array([max_acc1, max_acc2, ...]))
                                   }
        """
        self.start = start
        self.goal = goal
        self.max_iters = max_iters
        self.collision_checker = collision_checker
        self.state_limits = state_limits
        self.control_limits = control_limits

        # Initialize forward and backward trees with the start and goal nodes
        self.forward_tree = [self._create_node(state=start, parent=None)]
        self.backward_tree = [self._create_node(state=goal, parent=None)]

        # Store the final planned path
        self.path = None

    def _create_node(self, state, parent):
        """
        Create a node for the tree.

        Args:
            state (tuple of np.ndarray): State as (position, velocity).
            parent (dict): Parent node.

        Returns:
            dict: Node information containing state, parent, and trajectory.
        """
        return {
            'state': state,
            'parent': parent,
            'trajectory': None  # Can store the trajectory segment leading to this node
        }

    def seed_trees(self):
        """
        Seed the forward and backward trees with maximum braking trajectories.
        """
        # Forward tree: Generate maximum braking trajectory from start to zero velocity
        max_braking_traj = self.generate_max_braking_trajectory(self.start)
        if max_braking_traj:
            final_state = max_braking_traj[-1]
            self.forward_tree.append(self._create_node(state=final_state, parent=self.forward_tree[-1]))

        # Backward tree: Generate maximum reverse-braking trajectory from goal to zero velocity
        max_reverse_braking_traj = self.generate_max_reverse_braking_trajectory(self.goal)
        if max_reverse_braking_traj:
            final_state = max_reverse_braking_traj[-1]
            self.backward_tree.append(self._create_node(state=final_state, parent=self.backward_tree[-1]))

    def generate_max_braking_trajectory(self, state):
        """
        Generate a maximum braking trajectory from the given state to zero velocity.

        Args:
            state (tuple of np.ndarray): Current state as (position, velocity).

        Returns:
            list of tuple of np.ndarray: List of trajectory points as (position, velocity).
        """
        position, velocity = state
        min_accel, max_accel = self.control_limits['acceleration']
        
        # Calculate braking time for each dimension
        braking_time = np.abs(velocity / min_accel)
        t_stop = np.max(braking_time)  # Use the maximum time to ensure all dimensions stop

        # Time step
        dt = 0.1  # Can be adjusted
        num_steps = int(np.ceil(t_stop / dt))
        trajectory = []
        
        for step in range(1, num_steps + 1):
            t = step * dt
            acc = -min_accel  # Maximum braking acceleration
            # Ensure acceleration does not exceed limits
            acc = np.clip(acc, min_accel, max_accel)
            new_velocity = velocity + acc * dt
            new_velocity = np.maximum(new_velocity, np.zeros_like(new_velocity))  # Ensure velocity is non-negative
            new_position = position + velocity * dt + 0.5 * acc * (dt ** 2)
            trajectory.append((new_position.copy(), new_velocity.copy()))
            velocity = new_velocity
            position = new_position
        
        return trajectory

    def generate_max_reverse_braking_trajectory(self, state):
        """
        Generate a maximum reverse-braking trajectory from the given state to zero velocity.

        Args:
            state (tuple of np.ndarray): Current state as (position, velocity).

        Returns:
            list of tuple of np.ndarray: List of trajectory points as (position, velocity).
        """
        position, velocity = state
        min_accel, max_accel = self.control_limits['acceleration']
        
        # Reverse acceleration is positive maximum acceleration
        accel = max_accel
        braking_time = np.abs(velocity / accel)
        t_stop = np.max(braking_time)

        dt = 0.1  # Can be adjusted
        num_steps = int(np.ceil(t_stop / dt))
        trajectory = []
        
        for step in range(1, num_steps + 1):
            t = step * dt
            acc = accel  # Maximum acceleration
            acc = np.clip(acc, min_accel, max_accel)
            new_velocity = velocity + acc * dt
            new_velocity = np.minimum(new_velocity, np.zeros_like(new_velocity))  # Ensure velocity is non-positive
            new_position = position + velocity * dt + 0.5 * acc * (dt ** 2)
            trajectory.append((new_position.copy(), new_velocity.copy()))
            velocity = new_velocity
            position = new_position
        
        return trajectory

    def sample_random_state(self):
        """
        Sample a random zero-velocity state within the feasible set.

        Returns:
            tuple of np.ndarray: Random state as (position, velocity).
        """
        min_pos, max_pos = self.state_limits['position']
        # Sample position within limits
        position = np.array([random.uniform(min_p, max_p) for min_p, max_p in zip(min_pos, max_pos)])
        # Velocity is zero
        velocity = np.zeros_like(position)
        return (position, velocity)

    def construct_trajectory(self, state_from, state_to):
        """
        Construct a dynamically feasible trajectory between two states using a trapezoidal velocity profile.

        Args:
            state_from (tuple of np.ndarray): Starting state as (position, velocity).
            state_to (tuple of np.ndarray): Target state as (position, velocity).

        Returns:
            list of tuple of np.ndarray: List of trajectory points if successful, else None.
        """
        pos_from, vel_from = state_from
        pos_to, vel_to = state_to
        min_accel, max_accel = self.control_limits['acceleration']

        # Calculate differences
        delta_pos = pos_to - pos_from
        delta_vel = vel_to - vel_from

        # Calculate required acceleration
        # Simplified as uniform acceleration for each dimension
        acc = delta_vel / self.max_iters  # Assuming uniform acceleration

        # Check if acceleration is within limits
        if not np.all(acc >= min_accel) or not np.all(acc <= max_accel):
            return None

        # Generate trajectory
        trajectory = []
        dt = 0.1  # Time step, can be adjusted
        current_pos = pos_from.copy()
        current_vel = vel_from.copy()

        for step in range(1, self.max_iters + 1):
            current_acc = acc  # Uniform acceleration
            new_vel = current_vel + current_acc * dt
            new_pos = current_pos + current_vel * dt + 0.5 * current_acc * (dt ** 2)

            # Check if new position and velocity are within limits
            if not self._is_within_limits(new_pos, new_vel):
                return None

            trajectory.append((new_pos.copy(), new_vel.copy()))
            current_pos = new_pos
            current_vel = new_vel

        return trajectory

    def _is_within_limits(self, position, velocity):
        """
        Check if the given position and velocity are within the defined limits.

        Args:
            position (np.ndarray): Position vector.
            velocity (np.ndarray): Velocity vector.

        Returns:
            bool: True if within limits, False otherwise.
        """
        pos_min, pos_max = self.state_limits['position']
        vel_min, vel_max = self.state_limits['velocity']

        if not np.all(position >= pos_min) or not np.all(position <= pos_max):
            return False
        if not np.all(velocity >= vel_min) or not np.all(velocity <= vel_max):
            return False
        return True

    def extend_tree(self, tree, other_tree):
        """
        Extend the given tree by sampling and connecting states.

        Args:
            tree (list of dict): The tree to extend.
            other_tree (list of dict): The other tree to check for connections.
        """
        # Sample a random state
        random_state = self.sample_random_state()
        # Select a random node from the tree
        node = random.choice(tree)
        # Attempt to construct a trajectory from the node's state to the random state
        traj = self.construct_trajectory(node['state'], random_state)
        if traj:
            midpoint = traj[len(traj) // 2]
            new_node = self._create_node(state=midpoint, parent=node)
            tree.append(new_node)

    def check_for_connection(self):
        """
        Check if the forward and backward trees can be connected.

        Returns:
            list of tuple of np.ndarray: Combined trajectory if connection is found, else None.
        """
        # Iterate through nodes in forward and backward trees to find a connection
        for f_node in self.forward_tree:
            for b_node in self.backward_tree:
                traj = self.construct_trajectory(f_node['state'], b_node['state'])
                if traj:
                    # Combine trajectories
                    full_traj = self._combine_trajectories(f_node, traj, b_node)
                    return full_traj
        return None

    def _combine_trajectories(self, f_node, traj, b_node):
        """
        Combine forward trajectory, connecting trajectory, and backward trajectory.

        Args:
            f_node (dict): Node from the forward tree.
            traj (list of tuple of np.ndarray): Connecting trajectory.
            b_node (dict): Node from the backward tree.

        Returns:
            list of tuple of np.ndarray: Combined full trajectory.
        """
        # For simplicity, return the connecting trajectory
        # In practice, you would concatenate the forward path to the connecting trajectory
        # and then to the backward path (reversed)
        return traj  # Needs proper implementation based on application

    def check_collision(self, trajectory):
        """
        Check for collisions along the trajectory using the collision checker.

        Args:
            trajectory (list of tuple of np.ndarray): List of trajectory points.

        Returns:
            bool: True if collision-free, False otherwise.
        """
        for state in trajectory:
            if self.collision_checker(state):
                return False
        return True

    def shortcut_path(self, path):
        """
        Apply shortcutting to smooth the path.

        Args:
            path (list of tuple of np.ndarray): Original path.

        Returns:
            list of tuple of np.ndarray: Smoothed path.
        """
        # Placeholder for shortcutting implementation
        # Currently returns the original path
        return path

    def remove_infeasible_edge(self, tree, trajectory):
        """
        Remove the infeasible edge from the tree.

        Args:
            tree (list of dict): The tree from which to remove the edge.
            trajectory (list of tuple of np.ndarray): The trajectory with the infeasible edge.
        """
        # Placeholder for removing infeasible edges
        # Implementation depends on how edges are stored and managed
        pass

    def plan(self):
        """
        Plan the trajectory from the start to the goal state.

        Returns:
            list of tuple of np.ndarray: Planned trajectory if successful, else None.
        """
        # Seed the trees with maximum braking trajectories
        self.seed_trees()

        # Main planning loop
        for _ in range(self.max_iters):
            # Extend the forward tree
            self.extend_tree(self.forward_tree, self.backward_tree)

            # Extend the backward tree
            self.extend_tree(self.backward_tree, self.forward_tree)

            # Check for connections between trees
            trajectory = self.check_for_connection()
            if trajectory:
                # Check if the trajectory is collision-free
                if self.check_collision(trajectory):
                    # Apply path smoothing
                    smoothed_path = self.shortcut_path(trajectory)
                    self.path = smoothed_path
                    return smoothed_path
                else:
                    # Remove infeasible edges from both trees
                    self.remove_infeasible_edge(self.forward_tree, trajectory)
                    self.remove_infeasible_edge(self.backward_tree, trajectory)
        # Return None if planning was unsuccessful
        return None
