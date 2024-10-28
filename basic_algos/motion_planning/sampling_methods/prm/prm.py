import numpy as np
import networkx as nx

from scipy.spatial import cKDTree

class PRM:
    def __init__(self, num_samples, k_neighbors, collision_checker, sampling_area):
        self.num_samples = num_samples  # Number of samples to generate
        self.k_neighbors = k_neighbors  # Number of nearest neighbors to consider for connections
        self.collision_checker = collision_checker  # Function to check collisions between two points
        self.sampling_area = sampling_area  # Sampling area ((x_min, x_max), (y_min, y_max))
        self.nodes = []  # List of nodes (configurations) in the roadmap
        self.roadmap = nx.Graph()  # The roadmap graph

    def sample_free(self):
        """Sample a random configuration in the free space."""
        x_min, x_max = self.sampling_area[0]
        y_min, y_max = self.sampling_area[1]
        while True:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            config = (x, y)
            if self.collision_checker(config, config):
                return config  # Return the configuration if it's in free space

    def construct_roadmap(self):
        """Construct the probabilistic roadmap."""
        # Phase 1: Node generation
        for _ in range(self.num_samples):
            q = self.sample_free()
            self.nodes.append(q)
            self.roadmap.add_node(q)

        # Build a KD-tree for efficient neighbor search
        nodes_array = np.array(self.nodes)
        kdtree = cKDTree(nodes_array)

        # Phase 2: Edge addition
        for node in self.nodes:
            # Find k nearest neighbors
            distances, indices = kdtree.query(node, k=self.k_neighbors + 1)
            for idx in indices[1:]:  # Exclude the node itself
                neighbor = tuple(nodes_array[idx])
                # Attempt to connect node and neighbor
                if self.collision_checker(node, neighbor):
                    distance = np.linalg.norm(np.array(node) - np.array(neighbor))
                    self.roadmap.add_edge(node, neighbor, weight=distance)

    def query(self, start, goal):
        """Find a path from start to goal using the roadmap."""
        if not self.collision_checker(start, start) or not self.collision_checker(goal, goal):
            return None  # Start or goal is in collision

        # Connect start and goal to the roadmap
        self.roadmap.add_node(start)
        self.roadmap.add_node(goal)

        # Build a KD-tree for the existing nodes
        nodes_array = np.array(self.nodes)
        kdtree = cKDTree(nodes_array)

        # Connect start to nearest neighbors
        distances, indices = kdtree.query(start, k=self.k_neighbors)
        for idx in indices:
            neighbor = tuple(nodes_array[idx])
            if self.collision_checker(start, neighbor):
                distance = np.linalg.norm(np.array(start) - np.array(neighbor))
                self.roadmap.add_edge(start, neighbor, weight=distance)

        # Connect goal to nearest neighbors
        distances, indices = kdtree.query(goal, k=self.k_neighbors)
        for idx in indices:
            neighbor = tuple(nodes_array[idx])
            if self.collision_checker(goal, neighbor):
                distance = np.linalg.norm(np.array(goal) - np.array(neighbor))
                self.roadmap.add_edge(goal, neighbor, weight=distance)

        # Use Dijkstra's algorithm to find the shortest path
        try:
            path = nx.shortest_path(self.roadmap, source=start, target=goal, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None  # No path found