import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree

class KinodynamicRRT:
    def __init__(self, start, goal, obstacle_free, max_iters, state_limits, u_set, dynamics_model, control_duration, dt, goal_threshold):
        """
        Initialize the Kinodynamic RRT.
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacle_free = obstacle_free
        self.max_iters = max_iters
        self.state_limits = state_limits
        self.u_set = u_set
        self.dynamics_model = dynamics_model
        self.control_duration = control_duration
        self.dt = dt
        self.goal_threshold = goal_threshold
        self.tree = [self.start]
        self.parent = {tuple(self.start): None}
        self.controls = {tuple(self.start): None}
        self.all_edges = []
        self.num_nodes = 1

        # Compute weights to normalize each dimension to [-1, 1]
        self.weights = np.array([2 / (limit[1] - limit[0]) for limit in self.state_limits])

        # Initialize KDTree with the start state
        self.kd_tree = KDTree([self.apply_weights(self.start)])

    def apply_weights(self, state):
        """Apply weights to normalize the state."""
        return (state - np.array([limit[0] for limit in self.state_limits])) * self.weights - 1

    def sample_state(self, goal_bias=0.1):
        """Randomly sample a state within the state limits or with a goal bias."""
        if np.random.rand() < goal_bias:
            return self.goal
        else:
            return np.array([np.random.uniform(*limit) for limit in self.state_limits])

    def nearest(self, state):
        """Find the nearest state in the tree to the given state using KDTree."""
        weighted_state = self.apply_weights(state)
        _, nearest_idx = self.kd_tree.query(weighted_state)
        return self.tree[nearest_idx]

    def steer(self, state, control, method='euler'):
        """
        Propagate the state using the given control over the fixed control duration.
        """
        num_steps = int(self.control_duration / self.dt)
        new_state = state
        for _ in range(num_steps):
            new_state = self.dynamics_model.step(new_state, control, self.dt, method)
        return new_state
    
    def expand(self, x_nearest, x_target, integration_method='euler'):
        """
        Expand the tree toward x_target from x_nearest using kinodynamic constraints.
        Attempts to reach x_target by testing various control inputs and selecting
        the one that moves closest to x_target.
        
        Returns a status ('Reached', 'Advanced', 'Trapped') and the new state.
        """
        closest_new_node = None
        closest_distance = float('inf')
        closest_control = None

        # Iterate over all control inputs to find the best expansion towards x_target
        for control in self.u_set:
            x_new = self.steer(x_nearest, control, method=integration_method)

            if self.obstacle_free(x_nearest, x_new):
                distance = np.linalg.norm(self.apply_weights(x_new) - self.apply_weights(x_target))

                if distance < closest_distance:
                    closest_new_node = x_new
                    closest_distance = distance
                    closest_control = control

        if closest_new_node is not None:
            # Add the closest new node to the tree
            self.tree.append(closest_new_node)
            self.parent[tuple(closest_new_node)] = tuple(x_nearest)
            self.controls[tuple(closest_new_node)] = closest_control
            self.all_edges.append((x_nearest, closest_new_node))
            self.num_nodes += 1

            # Update the KDTree for efficient nearest-neighbor search
            self.kd_tree = KDTree([self.apply_weights(node) for node in self.tree])

            # Determine if the target has been reached or advanced toward
            if closest_distance <= self.goal_threshold:
                return 'Reached', closest_new_node
            else:
                return 'Advanced', closest_new_node

        # If no feasible path found
        return 'Trapped', x_nearest

    def plan(self, integration_method='euler', goal_bias=0.1):
        """
        Execute the Kinodynamic RRT planning algorithm.
        """
        for _ in tqdm(range(self.max_iters)):
            # Sample a random state with a goal bias
            x_rand = self.sample_state(goal_bias=goal_bias)
            
            # Find the nearest node in the tree to the random sample
            x_nearest = self.nearest(x_rand)
            
            # Use expand to attempt to move towards x_rand
            status, x_new = self.expand(x_nearest, x_rand, integration_method=integration_method)
            
            # If the new node is added successfully (status is 'Reached' or 'Advanced')
            if status != 'Trapped':
                # Check if the new node is within the goal threshold
                if np.linalg.norm(x_new - self.goal) <= self.goal_threshold:
                    if self.obstacle_free(x_new, self.goal):
                        # Add goal to the tree and link it to the last new node
                        self.tree.append(self.goal)
                        self.parent[tuple(self.goal)] = tuple(x_new)
                        self.controls[tuple(self.goal)] = None  # No control needed for the goal
                        self.all_edges.append((x_new, self.goal))
                        self.num_nodes += 1
                        return self.reconstruct_path(self.goal)

        return None  # Return None if no path is found within max_iters

    def reconstruct_path(self, end_state):
        """Reconstruct the path from start to the given end_state."""
        path = []
        state = tuple(end_state)
        while state is not None:
            path.append(np.array(state))
            state = self.parent.get(state)
        path.reverse()
        return path
