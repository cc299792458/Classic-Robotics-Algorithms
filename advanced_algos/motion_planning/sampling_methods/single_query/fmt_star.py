import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree

class FMTStar:
    def __init__(self, start, goal, obstacle_free, max_iters, connection_radius, sampling_range, num_samples):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacle_free = obstacle_free  # Function to check if the path between two points is collision-free
        self.max_iters = max_iters
        self.connection_radius = connection_radius
        self.sampling_range = sampling_range  # Sampling range as ((x_min, x_max), (y_min, y_max), ...)
        self.num_samples = num_samples

        # Initialize nodes and trees
        self.samples = self._generate_samples()
        self.tree = [self.start]
        self.parent = {self.start: None}
        self.cost = {self.start: 0.0}
        self.V_open = {self.start}
        self.V_unvisited = set(self.samples) - {self.start}
        self.V_closed = set()
        
        # KDTree for nearest-neighbor search
        self.kd_tree = KDTree(list(self.samples))

    def _generate_samples(self):
        """Generate random samples within the specified sampling range."""
        samples = [self.start, self.goal]
        for _ in range(self.num_samples - 2):  # Exclude start and goal from samples
            point = tuple(np.random.uniform(low, high) for low, high in self.sampling_range)
            samples.append(point)
        return samples

    def _get_neighbors(self, node):
        """Find neighbors of the node within the connection radius using KDTree."""
        indices = self.kd_tree.query_ball_point(node, self.connection_radius)
        neighbors = [tuple(self.kd_tree.data[i]) for i in indices if tuple(self.kd_tree.data[i]) in self.V_unvisited]
        return neighbors

    def _find_optimal_connection(self, x, neighbors):
        """Find the optimal node in V_open to connect to x."""
        min_cost = float('inf')
        best_parent = None
        for y in neighbors:
            if y in self.V_open and self.obstacle_free(y, x):
                cost = self.cost[y] + np.linalg.norm(np.array(y) - np.array(x))
                if cost < min_cost:
                    min_cost = cost
                    best_parent = y
        return best_parent, min_cost

    def plan(self, show_progress=True):
        """Run the FMT* algorithm to find an optimal path from start to goal."""
        iterator = tqdm(range(self.max_iters), desc="FMT* Planning Progress") if show_progress else range(self.max_iters)
        
        for _ in iterator:
            # Select the node with the lowest cost in V_open
            if not self.V_open:
                break  # No path found
            
            z = min(self.V_open, key=lambda node: self.cost[node])

            if z == self.goal:
                return self.reconstruct_path(self.goal)

            # Find neighbors of z in V_unvisited
            neighbors = self._get_neighbors(z)

            for x in neighbors:
                # Get neighbors of x in V_open and find the locally optimal connection
                y, cost = self._find_optimal_connection(x, neighbors)
                
                if y is not None:
                    # Add x to the tree with y as its parent
                    self.tree.append(x)
                    self.parent[x] = y
                    self.cost[x] = cost
                    self.V_open.add(x)
                    self.V_unvisited.remove(x)
                    
            # Move z from V_open to V_closed
            self.V_open.remove(z)
            self.V_closed.add(z)

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
