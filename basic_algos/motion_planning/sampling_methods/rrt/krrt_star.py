import numpy as np

from tqdm import tqdm
from scipy.spatial import KDTree  # Import KDTree for nearest neighbor search

class kRRTStar:
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range, dimension=2, num_nearest_neighbors=5):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacle_free = obstacle_free  # Function to check if the path between two points is collision-free
        self.max_iters = max_iters
        self.delta_distance = delta_distance  # Maximum extension distance
        self.sampling_range = sampling_range  # Sampling range as ((x_min, x_max), (y_min, y_max))
        self.dimension = dimension  # Dimension of the space, default is 2D
        self.num_nearest_neighbors = num_nearest_neighbors  # Number of nearest neighbors to consider, default is 5
        self.tree = [self.start]  # Initialize tree with the start node
        self.parent = {self.start: None}  # Dictionary to store parent of each node
        self.cost = {self.start: 0.0}  # Cost from start to each node
        self.all_nodes = []  # List to store nodes in the order they are added
        self.all_edges = []  # List to store all edges for visualization
        self.num_nodes = 1  # Initialize number of nodes with the start node

        self.kd_tree = KDTree([self.start])  # Initialize KDTree with start node

    def sample(self):
        """Randomly sample a point within the sampling range."""
        x_min, x_max = self.sampling_range[0]
        y_min, y_max = self.sampling_range[1]
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return (x, y)

    def nearest(self, point):
        """Find the nearest node in the KDTree to the given point."""
        _, nearest_idx = self.kd_tree.query(point)
        return self.tree[nearest_idx]

    def find_k_nearest(self, point, k):
        """Find the k-nearest nodes in the KDTree to the given point."""
        # Ensure k does not exceed the number of nodes in the tree
        k = min(k, len(self.tree))  # Prevent out of range issues

        if k == 1:
            # If k is 1, return the nearest node in a list
            _, nearest_idx = self.kd_tree.query(point)
            nearest_nodes = [self.tree[nearest_idx]]  # Wrap the result in a list
        else:
            _, nearest_indices = self.kd_tree.query(point, k)
            nearest_nodes = [self.tree[idx] for idx in nearest_indices]

        return nearest_nodes

    def steer(self, from_node, to_point):
        """
        Return a new node in the direction from 'from_node' to 'to_point',
        at a distance limited by 'delta_distance'.
        """
        from_node = np.array(from_node)
        to_point = np.array(to_point)
        direction_vector = to_point - from_node
        distance = np.linalg.norm(direction_vector)
        if distance <= self.delta_distance:
            return tuple(to_point)
        else:
            direction = direction_vector / distance
            new_node = from_node + direction * self.delta_distance
            return tuple(new_node)

    def plan(self, show_progress=True):
        """Run the k-RRT* algorithm to find an optimal path from start to goal."""
        iterator = tqdm(range(self.max_iters), desc="k-RRT* Planning Progress") if show_progress else range(self.max_iters)
        for i in iterator:
            x_sample = self.sample()
            x_nearest = self.nearest(x_sample)
            x_new = self.steer(x_nearest, x_sample)
            if self.obstacle_free(x_nearest, x_new):
                # Compute the cost to reach x_new via x_nearest
                c_min = self.cost[x_nearest] + np.linalg.norm(np.array(x_new) - np.array(x_nearest))
                x_min = x_nearest

                # Find the k nearest nodes using the KDTree
                k_near_set = self.find_k_nearest(x_new, self.num_nearest_neighbors)

                # Choose the node that offers the lowest cost to x_new
                for x_near in k_near_set:
                    if self.obstacle_free(x_near, x_new):
                        c = self.cost[x_near] + np.linalg.norm(np.array(x_new) - np.array(x_near))
                        if c < c_min:
                            c_min = c
                            x_min = x_near

                # Add x_new to the tree
                self.tree.append(x_new)
                self.parent[x_new] = x_min
                self.cost[x_new] = c_min
                self.all_edges.append((x_min, x_new))
                self.all_nodes.append(x_new)
                self.num_nodes += 1  # Increment the number of nodes

                # Rebuild the KDTree with the new node
                self.kd_tree = KDTree(self.tree)

                # Rewire the tree
                for x_near in k_near_set:
                    if x_near == x_min:
                        continue
                    if self.obstacle_free(x_new, x_near):
                        c = self.cost[x_new] + np.linalg.norm(np.array(x_new) - np.array(x_near))
                        if c < self.cost[x_near]:
                            # Update parent and cost
                            old_parent = self.parent[x_near]
                            self.parent[x_near] = x_new
                            self.cost[x_near] = c
                            # Update edge information
                            self.all_edges.remove((old_parent, x_near))
                            self.all_edges.append((x_new, x_near))

                # Check if the goal has been reached
                if np.linalg.norm(np.array(x_new) - np.array(self.goal)) <= self.delta_distance:
                    if self.obstacle_free(x_new, self.goal):
                        # Check if the cost via x_new is lower
                        c = self.cost[x_new] + np.linalg.norm(np.array(self.goal) - np.array(x_new))
                        if self.goal not in self.cost or c < self.cost[self.goal]:
                            if self.goal in self.parent:
                                # Remove old connection to goal
                                old_parent = self.parent[self.goal]
                                self.all_edges.remove((old_parent, self.goal))
                            self.tree.append(self.goal)
                            self.parent[self.goal] = x_new
                            self.cost[self.goal] = c
                            self.all_edges.append((x_new, self.goal))
                            self.all_nodes.append(self.goal)
                            self.num_nodes += 1  # Increment for goal node

        # If goal was added to the tree, reconstruct the path
        if self.goal in self.parent:
            return self.reconstruct_path(self.goal)
        else:
            return None  # Return None if no path is found within max_iters

    def reconstruct_path(self, end_node):
        """Reconstruct the path from start to the given end_node."""
        path = []
        node = end_node
        while node is not None:
            path.append(node)
            node = self.parent.get(node)
        path.reverse()
        return path
