import numpy as np

from tqdm import tqdm

class KinodynamicRRT:
    def __init__(self, start, goal, obstacle_free, max_iters, state_limits, u_limits, dynamics_model, control_duration, dt):
        """
        Initialize the Kinodynamic RRT.

        Params:
        - start: Initial state (could be any format compatible with dynamics model).
        - goal: Target state.
        - obstacle_free: Function to check if a transition between two states is collision-free.
        - max_iters: Maximum number of iterations for the RRT algorithm.
        - state_limits: Limits for each dimension of the state.
        - u_limits: Control input limits.
        - dynamics_model: An instance of a dynamics model class with a .step() method.
        - control_duration: Fixed duration for control propagation.
        - dt: Time step for integration.
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacle_free = obstacle_free
        self.max_iters = max_iters
        self.state_limits = state_limits
        self.u_limits = u_limits
        self.dynamics_model = dynamics_model
        self.control_duration = control_duration
        self.dt = dt
        self.tree = [self.start]
        self.parent = {tuple(self.start): None}
        self.controls = {tuple(self.start): None}
        self.all_edges = []
        self.num_nodes = 1

    def sample_state(self, goal_bias=0.1):
        """Randomly sample a state within the state limits or with a goal bias."""
        if np.random.rand() < goal_bias:
            return self.goal
        else:
            return np.array([np.random.uniform(*limit) for limit in self.state_limits])
        
    def sample_control(self):
        """Randomly sample control inputs within the control limits."""
        return np.array([np.random.uniform(*limit) for limit in self.u_limits])

    def nearest(self, tree, state, weights=None):
        """Find the nearest state in the tree to the given state."""
        weights = np.ones(len(state)) if weights is None else weights
        distances = [
            np.linalg.norm((np.array(node) - state) * weights) for node in tree
        ]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]

    def propagate(self, state, control, method='euler'):
        """
        Propagate the state using the given control over the fixed control duration.

        Params:
        - state: Current state.
        - control: Control input to apply.
        - method: Integration method ('euler' or 'rk4').

        Returns:
        - New state after propagation.
        """
        num_steps = int(self.control_duration / self.dt)
        new_state = state
        for _ in range(num_steps):
            new_state = self.dynamics_model.step(new_state, control, self.dt, method)
        return new_state

    def plan(self, integration_method='euler', goal_threshold=2.0):
        """
        Execute the Kinodynamic RRT planning algorithm.

        Params:
        - integration_method: Method for integration (default is 'euler').
        - goal_threshold: Threshold distance to consider the goal reached.

        Returns:
        - A list representing the path from start to goal, if found.
        """
        for _ in tqdm(range(self.max_iters)):
            x_rand = self.sample_state()
            x_nearest = self.nearest(self.tree, x_rand)
            control = self.sample_control()
            x_new = self.propagate(x_nearest, control, method=integration_method)

            if self.obstacle_free(x_nearest, x_new) and not any(np.allclose(x_new, node) for node in self.tree):
                self.tree.append(x_new)
                self.parent[tuple(x_new)] = tuple(x_nearest)
                self.controls[tuple(x_new)] = control
                self.all_edges.append((x_nearest, x_new))
                self.num_nodes += 1

                if np.linalg.norm(x_new[:len(self.goal)] - self.goal) <= goal_threshold:
                    return self.reconstruct_path(x_new)
                
        return None

    def reconstruct_path(self, end_state):
        """Reconstruct the path from start to the given end_state."""
        path = []
        state = tuple(end_state)
        while state is not None:
            path.append(np.array(state))
            state = self.parent.get(state)
        path.reverse()
        return path
