import numpy as np

from scipy.spatial import cKDTree

class RRT:
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacle_free = obstacle_free  # Function to check if the path between two points is collision-free
        self.max_iters = max_iters
        self.delta_distance = delta_distance  # Maximum extension distance
        self.sampling_range = sampling_range  # Sampling range as ((x_min, x_max), (y_min, y_max))
        self.tree = [self.start]  # Initialize tree with the start node
        self.parent = {self.start: None}  # Dictionary to store parent of each node
        self.all_nodes = []  # List to store nodes in the order they are added
        self.all_edges = []  # List to store all edges for visualization
        self.num_nodes = 1  # Initialize number of nodes with the start node

        # Initialize KD Tree for efficient nearest-neighbor search
        self.kd_tree = cKDTree([self.start])

    def sample(self):
        """Randomly sample a point within the sampling range."""
        x_min, x_max = self.sampling_range[0]
        y_min, y_max = self.sampling_range[1]
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return (x, y)

    def nearest(self, tree, point, kd_tree):
        """Find the nearest node in the specified tree to the given point using the provided KD Tree."""
        distance, idx = kd_tree.query(point)
        return tree[idx]

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
        
    def extend(self, tree, parent, x_nearest, x_target, is_goal_tree=False):
        """
        Extend the tree towards x_target from x_nearest.
        Returns a status ('Reached', 'Advanced', 'Trapped') and the new node.
        """
        x_new = self.steer(x_nearest, x_target)
        if self.obstacle_free(x_nearest, x_new):
            tree.append(x_new)
            parent[x_new] = x_nearest
            self.all_edges.append((x_nearest, x_new))
            self.num_nodes += 1  # Increment the number of nodes

            # Update KD tree with the new node
            if not is_goal_tree:
                self.kd_tree = cKDTree(tree)
            else:
                self.kd_tree_goal = cKDTree(tree)

            if np.linalg.norm(np.array(x_new) - np.array(x_target)) < self.delta_distance:
                return 'Reached', x_new
            else:
                return 'Advanced', x_new
        return 'Trapped', x_nearest

    def plan(self):
        """Run the RRT algorithm to find a path from start to goal."""
        for i in range(self.max_iters):
            x_sample = self.sample()
            x_nearest = self.nearest(self.tree, x_sample, self.kd_tree)
            status, x_new = self.extend(self.tree, self.parent, x_nearest, x_sample)
            if status != 'Trapped':
                self.all_nodes.append(x_new)

                # Check if the goal has been reached
                if np.linalg.norm(np.array(x_new) - np.array(self.goal)) <= self.delta_distance:
                    if self.obstacle_free(x_new, self.goal):
                        self.tree.append(self.goal)
                        self.parent[self.goal] = x_new
                        self.all_edges.append((x_new, self.goal))
                        self.all_nodes.append(self.goal)
                        self.num_nodes += 1  # Increment for goal node
                        return self.reconstruct_path(self.goal)
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
