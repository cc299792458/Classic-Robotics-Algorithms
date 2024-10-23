import numpy as np

class RRT:
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, goal_sample_rate, sampling_range):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacle_free = obstacle_free  # Function to check if a motion is collision-free
        self.max_iters = max_iters
        self.delta_distance = delta_distance  # Step size for each extension
        self.goal_sample_rate = goal_sample_rate  # Probability of sampling the goal
        self.sampling_range = sampling_range  # Sampling range (tuple indicating the range in each dimension)
        self.tree = [tuple(self.start)]  # Initialize tree with the start node
        self.parent = {tuple(self.start): None}  # Track parents to reconstruct path
        self.all_edges = []  # To store all edges for visualization
        self.num_nodes = 1  # Start with the initial node

    def sample(self):
        """Randomly sample a point in space, biased towards the goal with some probability."""
        if np.random.rand() > self.goal_sample_rate:
            return tuple(np.random.rand(2) * np.array(self.sampling_range))  # Random point in the space
        return tuple(self.goal)  # Sample the goal with some probability

    def nearest(self, point):
        """Find the nearest node in the tree to the sampled point."""
        distances = [np.linalg.norm(np.array(node) - point) for node in self.tree]
        nearest_idx = np.argmin(distances)
        return np.array(self.tree[nearest_idx])

    def local_planner(self, x_nearest, x_sample):
        """Generate a new point towards the sample within the step size delta_distance."""
        assert np.linalg.norm(x_sample - x_nearest) > 0, 'Distance should be larger than 0'
        direction = (x_sample - x_nearest) / np.linalg.norm(x_sample - x_nearest)
        x_new = x_nearest + direction * self.delta_distance
        return tuple(x_new)

    def reconstruct_path(self, x_new):
        """Reconstruct the path from start to goal."""
        path = [x_new]
        while x_new is not None:  # Ensure we stop when we reach the start node (which has None as parent)
            x_new = self.parent.get(tuple(x_new), None)
            if x_new is not None:
                path.append(x_new)
        return path[::-1]  # Return reversed path

    def plan(self):
        """Run the RRT algorithm to find a path to the goal."""
        for i in range(self.max_iters):
            x_sample = self.sample()
            x_nearest = self.nearest(np.array(x_sample))
            x_new = self.local_planner(x_nearest, np.array(x_sample))
            if self.obstacle_free(x_nearest, x_new):  # Check for collision-free motion
                self.tree.append(x_new)
                self.parent[x_new] = tuple(x_nearest)
                self.all_edges.append((x_nearest, x_new))  # Store the edge for tree visualization
                self.num_nodes += 1  # Increment node count

                # Check if we've reached the goal region
                if np.linalg.norm(np.array(x_new) - self.goal) < self.delta_distance and self.obstacle_free(np.array(x_new), self.goal):
                    return self.reconstruct_path(x_new)

        return None  # Return failure if no path is found within max_iters