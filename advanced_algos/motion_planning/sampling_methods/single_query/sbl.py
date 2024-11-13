import numpy as np

from tqdm import tqdm
from scipy.spatial import KDTree

class SBL:
    def __init__(self, start, goal, obstacle_free, max_iters, state_limits, goal_threshold):
        """
        Initialize the SBL algorithm.
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacle_free = obstacle_free
        self.max_iters = max_iters
        self.state_limits = state_limits
        self.goal_threshold = goal_threshold
        self.tree_a = [self.start]
        self.tree_b = [self.goal]
        self.parent_a = {tuple(self.start): None}
        self.parent_b = {tuple(self.goal): None}
        self.num_nodes = 2

        # Compute weights to normalize each dimension to [-1, 1]
        self.weights = np.array([2 / (limit[1] - limit[0]) for limit in self.state_limits])

    def apply_weights(self, state):
        """Apply weights to normalize the state."""
        return (state - np.array([limit[0] for limit in self.state_limits])) * self.weights - 1

    def sample_state(self):
        """Randomly sample a state within the state limits."""
        return np.array([np.random.uniform(*limit) for limit in self.state_limits])

    def nearest(self, tree, state):
        """Find the nearest state in the tree to the given state."""
        kd_tree = KDTree([self.apply_weights(node) for node in tree])
        _, nearest_idx = kd_tree.query(self.apply_weights(state))
        return tree[nearest_idx]

    def connect(self, x_nearest, x_new, tree, parent_dict):
        """
        Attempt to connect x_new to x_nearest in the given tree.
        If successful, add x_new to the tree and update parent_dict.
        """
        if self.obstacle_free(x_nearest, x_new):
            tree.append(x_new)
            parent_dict[tuple(x_new)] = tuple(x_nearest)
            return True
        return False

    def attempt_connection(self, tree_a, tree_b, parent_a, parent_b):
        """
        Attempt to connect tree_a and tree_b.
        """
        for x_a in tree_a:
            for x_b in tree_b:
                if self.obstacle_free(x_a, x_b):
                    # Connect two trees and return the path
                    return self.reconstruct_path(x_a, x_b, parent_a, parent_b)
        return None

    def reconstruct_path(self, x_a, x_b, parent_a, parent_b):
        """Reconstruct the path connecting the two trees."""
        path_a = []
        state = tuple(x_a)
        while state is not None:
            path_a.append(np.array(state))
            state = parent_a.get(state)
        path_a.reverse()

        path_b = []
        state = tuple(x_b)
        while state is not None:
            path_b.append(np.array(state))
            state = parent_b.get(state)

        return path_a + path_b

    def plan(self):
        """
        Execute the SBL planning algorithm.
        """
        for _ in tqdm(range(self.max_iters)):
            # Sample a random state
            x_rand = self.sample_state()

            # Extend tree A toward x_rand
            x_nearest_a = self.nearest(self.tree_a, x_rand)
            if self.connect(x_nearest_a, x_rand, self.tree_a, self.parent_a):
                # Try to connect tree B to the new node in tree A
                x_new_a = x_rand
                path = self.attempt_connection(self.tree_a, self.tree_b, self.parent_a, self.parent_b)
                if path:
                    return path

            # Extend tree B toward x_rand
            x_nearest_b = self.nearest(self.tree_b, x_rand)
            if self.connect(x_nearest_b, x_rand, self.tree_b, self.parent_b):
                # Try to connect tree A to the new node in tree B
                x_new_b = x_rand
                path = self.attempt_connection(self.tree_b, self.tree_a, self.parent_b, self.parent_a)
                if path:
                    return path

        return None  # Return None if no path is found within max_iters