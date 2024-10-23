import numpy as np
from scipy.integrate import solve_ivp

class KinodynamicRRT:
    def __init__(self, start, goal, obstacle_free, max_iters, sampling_range, u_limits, dt, max_time):
        self.start = start  # State: (x, y, vx, vy)
        self.goal = goal
        self.obstacle_free = obstacle_free  # Function to check path collision
        self.max_iters = max_iters
        self.sampling_range = sampling_range  # ((x_min, x_max), (y_min, y_max), (vx_min, vx_max), (vy_min, vy_max))
        self.u_limits = u_limits  # Control input limits: ((ux_min, ux_max), (uy_min, uy_max))
        self.dt = dt  # Time step for integration
        self.max_time = max_time  # Maximum propagation time
        self.tree = [self.start]
        self.parent = {self.start: None}
        self.controls = {self.start: None}  # Store control inputs leading to each state

    def sample_state(self):
        """Randomly sample a state within the sampling range."""
        x_min, x_max = self.sampling_range[0]
        y_min, y_max = self.sampling_range[1]
        vx_min, vx_max = self.sampling_range[2]
        vy_min, vy_max = self.sampling_range[3]
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        vx = np.random.uniform(vx_min, vx_max)
        vy = np.random.uniform(vy_min, vy_max)
        return (x, y, vx, vy)

    def nearest(self, tree, state):
        """Find the nearest node in the tree to the given state."""
        distances = [np.linalg.norm(np.array(node[:2]) - np.array(state[:2])) for node in tree]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]

    def sample_control(self):
        """Randomly sample control inputs within the control limits."""
        ux_min, ux_max = self.u_limits[0]
        uy_min, uy_max = self.u_limits[1]
        ux = np.random.uniform(ux_min, ux_max)
        uy = np.random.uniform(uy_min, uy_max)
        return (ux, uy)

    def propagate(self, state, control, time):
        """Propagate the state using the given control over the specified time."""
        def dynamics(t, s):
            x, y, vx, vy = s
            ux, uy = control
            return [vx, vy, ux, uy]

        t_span = [0, time]
        sol = solve_ivp(dynamics, t_span, state, t_eval=[time], max_step=self.dt)
        if sol.status == 0:
            new_state = tuple(sol.y[:, -1])
            return new_state
        else:
            return None

    def plan(self):
        """Run the Kinodynamic RRT algorithm to find a path from start to goal."""
        for i in range(self.max_iters):
            x_rand = self.sample_state()
            x_nearest = self.nearest(self.tree, x_rand)
            control = self.sample_control()
            time = np.random.uniform(0, self.max_time)
            x_new = self.propagate(x_nearest, control, time)
            if x_new and self.obstacle_free(x_nearest, x_new):
                self.tree.append(x_new)
                self.parent[x_new] = x_nearest
                self.controls[x_new] = control

                # Check if the goal has been reached within a threshold
                if np.linalg.norm(np.array(x_new[:2]) - np.array(self.goal[:2])) <= self.dt:
                    path = self.reconstruct_path(x_new)
                    return path
        return None  # Return None if no path is found within max_iters

    def reconstruct_path(self, end_state):
        """Reconstruct the path from start to the given end_state."""
        path = []
        state = end_state
        while state is not None:
            path.append(state)
            state = self.parent.get(state)
        path.reverse()
        return path
