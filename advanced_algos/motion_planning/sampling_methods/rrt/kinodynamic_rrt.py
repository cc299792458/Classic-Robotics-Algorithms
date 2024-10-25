import numpy as np

from tqdm import tqdm
from scipy.integrate import solve_ivp

class KinodynamicRRT:
    def __init__(self, start, goal, obstacle_free, max_iters, sampling_range, u_limits, dt, max_time):
        self.start = np.array(start)  # State: (x, y, vx, vy)
        self.goal = np.array(goal)
        self.obstacle_free = obstacle_free  # Function to check if the trajectory is collision-free
        self.max_iters = max_iters
        self.sampling_range = sampling_range  # Sampling ranges for (x, y, vx, vy)
        self.u_limits = u_limits  # Control input limits for (ux, uy)
        self.dt = dt  # Time step for integration
        self.max_time = max_time  # Maximum propagation time
        self.tree = [self.start]  # Initialize tree with the start state
        self.parent = {tuple(self.start): None}  # Dictionary to store parent of each state
        self.controls = {tuple(self.start): None}  # Dictionary to store control inputs
        self.all_edges = []  # List to store all edges for visualization
        self.num_nodes = 1  # Initialize number of nodes with the start state

    def sample_state(self, goal_bias=0.1):
        """Randomly sample a state within the sampling ranges or with a goal bias."""
        if np.random.rand() < goal_bias:
            return self.goal  # Bias towards goal
        else:
            x = np.random.uniform(*self.sampling_range[0])
            y = np.random.uniform(*self.sampling_range[1])
            vx = np.random.uniform(*self.sampling_range[2])
            vy = np.random.uniform(*self.sampling_range[3])
            return np.array([x, y, vx, vy])

    def nearest(self, tree, state, position_weight=1.0, velocity_weight=1.0):
        """
        Find the nearest state in the tree to the given state.
        The distance is computed with weighted and normalized position and velocity differences.
        
        :param tree: The current set of nodes (states) in the tree.
        :param state: The state for which we are finding the nearest neighbor.
        :param position_weight: Weight for the position difference.
        :param velocity_weight: Weight for the velocity difference.
        :return: The nearest state in the tree.
        """
        # Get the ranges for normalization
        x_range = self.sampling_range[0][1] - self.sampling_range[0][0]
        y_range = self.sampling_range[1][1] - self.sampling_range[1][0]
        vx_range = self.sampling_range[2][1] - self.sampling_range[2][0]
        vy_range = self.sampling_range[3][1] - self.sampling_range[3][0]

        distances = [
            position_weight * (np.linalg.norm((node[:2] - state[:2]) / np.array([x_range, y_range]))) +
            velocity_weight * (np.linalg.norm((node[2:] - state[2:]) / np.array([vx_range, vy_range])))
            for node in tree
        ]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]

    def sample_control(self):
        """Randomly sample control inputs within the control limits."""
        ux = np.random.uniform(*self.u_limits[0])
        uy = np.random.uniform(*self.u_limits[1])
        return np.array([ux, uy])

    def propagate(self, state, control, time):
        """Propagate the state using the given control over the specified time."""
        def dynamics(t, s):
            x, y, vx, vy = s
            ux, uy = control
            return [vx, vy, ux, uy]

        t_span = [0, time]
        sol = solve_ivp(dynamics, t_span, state, t_eval=[time], max_step=self.dt)
        if sol.status == 0:
            new_state = sol.y[:, -1]
            return new_state
        else:
            return None

    def plan(self):
        """Run the Kinodynamic RRT algorithm to find a path from start to goal."""
        for _ in tqdm(range(self.max_iters)):
            x_rand = self.sample_state()
            x_nearest = self.nearest(self.tree, x_rand)
            control = self.sample_control()
            time = np.random.uniform(self.dt, self.max_time)
            x_new = self.propagate(x_nearest, control, time)

            # Check if the new state is collision-free and not already in the tree
            if self.obstacle_free(x_nearest, x_new) and not any(np.allclose(x_new, node) for node in self.tree):
                self.tree.append(x_new)
                self.parent[tuple(x_new)] = tuple(x_nearest)
                self.controls[tuple(x_new)] = control
                self.all_edges.append((x_nearest, x_new))
                self.num_nodes += 1  # Increment the number of nodes

                # Check if the goal has been reached within a threshold
                if np.linalg.norm(x_new[:2] - self.goal[:2]) <= 2.0:
                    return self.reconstruct_path(x_new)
                
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
