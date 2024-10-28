import numpy as np
import networkx as nx

from scipy.spatial import cKDTree
from basic_algos.motion_planning.sampling_methods.prm import PRM

class LazyPRM(PRM):
    def __init__(self, num_samples, k_neighbors, collision_checker, sampling_area):
        super().__init__(num_samples, k_neighbors, collision_checker, sampling_area)
        self.lazy_edges = {}  # Store edges that are not yet checked for collision

    def construct_roadmap(self):
        """Construct the probabilistic roadmap without checking for collisions immediately."""
        # Phase 1: Node generation (same as PRM)
        for _ in range(self.num_samples):
            q = self.sample_free()
            self.nodes.append(q)
            self.roadmap.add_node(q)

        # Build a KD-tree for efficient neighbor search
        nodes_array = np.array(self.nodes)
        kdtree = cKDTree(nodes_array)

        # Phase 2: Edge addition (but don't check collisions yet)
        for node in self.nodes:
            # Find k nearest neighbors
            distances, indices = kdtree.query(node, k=self.k_neighbors + 1)
            for idx in indices[1:]:  # Exclude the node itself
                neighbor = tuple(nodes_array[idx])
                # Lazy PRM: Store edge but delay collision check
                distance = np.linalg.norm(np.array(node) - np.array(neighbor))
                self.lazy_edges[(node, neighbor)] = distance  # Save the edge distance without checking collision
                self.roadmap.add_edge(node, neighbor, weight=distance)

    def _check_and_remove_invalid_edges(self, path):
        """Perform collision checking on the edges in the path lazily."""
        valid = True
        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]
            edge = (node, next_node)

            # Check if this edge has not been checked yet
            if edge in self.lazy_edges or (next_node, node) in self.lazy_edges:
                # Perform collision check only when needed
                if not self.collision_checker(node, next_node):
                    # Remove the invalid edge from the roadmap
                    self.roadmap.remove_edge(node, next_node)
                    valid = False  # Path is invalid
                # Regardless of collision check result, remove the edge from lazy_edges
                self.lazy_edges.pop(edge, None)
                self.lazy_edges.pop((next_node, node), None)
        return valid

    def query(self, start, goal):
        """Find a path from start to goal, lazily checking collisions on edges."""
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
        except nx.NetworkXNoPath:
            return None  # No path found

        # Lazy PRM: Perform collision checking lazily on the found path
        if self._check_and_remove_invalid_edges(path):
            return path  # Return the valid path
        else:
            return None  # No valid path found after checking
