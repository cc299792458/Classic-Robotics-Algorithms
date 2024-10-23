import numpy as np

from basic_algos.motion_planning.sampling_methods.rrt.rrt import RRT

class RRTConnect(RRT):
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range):
        super().__init__(start, goal, obstacle_free, max_iters, delta_distance, 0.0, sampling_range)
        self.goal_tree = [tuple(self.goal)]  # Initialize goal tree
        self.goal_parent = {tuple(self.goal): None}  # Parent dictionary for the goal tree

    def sample(self):
        """Randomly sample a point in space"""
        return tuple(np.random.rand(2) * np.array(self.sampling_range))  # Random point in the space

    def nearest(self, tree, point):
        """Find the nearest node in the specified tree to the sampled point."""
        distances = [np.linalg.norm(np.array(node) - point) for node in tree]
        nearest_idx = np.argmin(distances)
        return np.array(tree[nearest_idx])

    def extend(self, tree, parent, x_nearest, x_sample):
        """Extend the tree towards the sampled point."""
        x_new = self.local_planner(x_nearest, np.array(x_sample))
        if self.obstacle_free(x_nearest, x_new):
            tree.append(x_new)
            parent[x_new] = tuple(x_nearest)
            return x_new
        return None

    def connect(self, tree, parent, x_sample):
        """Try to connect the tree to the sample point."""
        x_nearest = self.nearest(tree, np.array(x_sample))
        while True:
            x_new = self.extend(tree, parent, x_nearest, x_sample)
            x_nearest = x_new
            if x_new is None or np.linalg.norm(np.array(x_new) - x_sample) < self.delta_distance:
                break
        return x_nearest

    def plan(self):
        """Run the RRT-Connect algorithm to find a path to the goal."""
        for i in range(self.max_iters):
            # Sample a random point and extend the start tree
            x_sample = self.sample()
            x_nearest_start = self.nearest(self.tree, np.array(x_sample))
            x_new_start = self.extend(self.tree, self.parent, x_nearest_start, x_sample)

            if x_new_start is not None:
                # Try to connect the goal tree to the new node in the start tree
                x_new_goal = self.connect(self.goal_tree, self.goal_parent, x_new_start)

                # Check if the trees have connected
                if x_new_goal is not None:
                    return self.reconstruct_bidirectional_path(x_new_start, x_new_goal)

            # Swap trees (alternate the start and goal trees for balanced growth)
            self.tree, self.goal_tree = self.goal_tree, self.tree
            self.parent, self.goal_parent = self.goal_parent, self.parent

        return None  # Return failure if no path is found within max_iters

    def reconstruct_bidirectional_path(self, x_start, x_goal):
        """Reconstruct the path from start to goal using both trees."""
        path_start = self.reconstruct_path(x_start)  # Path from start tree
        path_goal = self.reconstruct_path(x_goal)  # Path from goal tree
        return path_start[:-1] + path_goal[::-1]  # Combine paths, removing duplicated node at connection point
