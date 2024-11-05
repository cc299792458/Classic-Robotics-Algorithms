import numpy as np

from tqdm import tqdm
from basic_algos.motion_planning.sampling_methods.rrt import kRRTStar

class InformedkRRTStar(kRRTStar):
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range, num_nearest_neighbors=33):
        super().__init__(start, goal, obstacle_free, max_iters, delta_distance, sampling_range, num_nearest_neighbors)
        self.best_path_length = float('inf')  # Initialize best path length as infinity

    def sample(self):
        """Sample a point within the informed region if a path has been found."""
        # Check if we have found an initial path to the goal
        if self.best_path_length < float('inf'):
            return self.informed_sample()
        else:
            return super().sample()  # Use the standard sampling method if no path is found
        
    def informed_sample(self):
        """
        Sample a point within the ellipsoidal region defined by the best path length.
        """
        # Compute ellipsoid parameters based on start, goal, and best path length
        c_max = self.best_path_length
        c_min = np.linalg.norm(np.array(self.goal) - np.array(self.start))
        center = (np.array(self.start) + np.array(self.goal)) / 2.0
        direction = np.array(self.goal) - np.array(self.start)

        # Define axis lengths for the ellipsoid
        r1 = c_max / 2.0  # Half of the best path length for the main axis
        rn = np.sqrt(c_max**2 - c_min**2) / 2.0  # Radius of the minor axes

        while True:  # Repeat until a valid sample within the sampling range is found
            # Generate a random point within a unit ball in the sampling space dimension
            random_point = self._sample_unit_ball(len(self.start))

            # Scale the random point to fit within the ellipsoid
            scaled_point = np.array([r1 * random_point[0]] + [rn * coord for coord in random_point[1:]])

            # Compute rotation matrix to align the ellipsoid with the direction vector
            rotation_matrix = self._compute_rotation_matrix(direction)

            # Apply the rotation and translation to transform the sample into the ellipsoid
            informed_sample = rotation_matrix.dot(scaled_point) + center

            # Check if the informed_sample is within the sampling range
            if all(self.sampling_range[i][0] <= informed_sample[i] <= self.sampling_range[i][1] for i in range(len(informed_sample))):
                return tuple(informed_sample)

    def _sample_unit_ball(self, dimension):
        """Generate a random point within a unit ball in the specified dimension using numpy."""
        point = np.random.normal(size=dimension)  # Generate normally distributed point
        point /= np.linalg.norm(point)  # Normalize to lie on the unit sphere
        radius = np.random.uniform(0, 1) ** (1.0 / dimension)  # Uniform radius scaling
        return point * radius  # Scale to lie within unit ball

    def _compute_rotation_matrix(self, direction):
        """
        Compute a rotation matrix that aligns the ellipsoid's main axis with the direction
        from start to goal.
        """
        dimension = len(direction)
        if dimension == 2:
            # 2D case: simple rotation matrix
            theta = np.arctan2(direction[1], direction[0])
            return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        elif dimension == 3:
            # 3D case: compute rotation using the cross-product
            direction = direction / np.linalg.norm(direction)
            z = np.array([0, 0, 1])
            v = np.cross(z, direction)
            s = np.linalg.norm(v)
            c = np.dot(z, direction)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + vx + (vx @ vx) * ((1 - c) / s**2 if s != 0 else 0)
        else:
            # Higher dimensions are not implemented; fall back to identity
            return np.eye(dimension)

    def _add_node_to_tree(self, x_new, x_min, c_min):
        """
        Override to update the best path length if we find a better path to the goal.
        """
        super()._add_node_to_tree(x_new, x_min, c_min)

        # If x_new is close to the goal, update best path length
        if np.linalg.norm(np.array(x_new) - np.array(self.goal)) <= self.delta_distance:
            self.best_path_length = min(self.best_path_length, c_min + np.linalg.norm(np.array(self.goal) - np.array(x_new)))

    def plan(self, show_progress=True):
        """Run the informed k-RRT* algorithm to find an optimal path from start to goal."""
        iterator = tqdm(range(self.max_iters), desc="Informed k-RRT* Planning Progress") if show_progress else range(self.max_iters)

        for i in iterator:
            x_sample = self.sample()
            x_nearest = self.nearest(x_sample)
            x_new = self.steer(x_nearest, x_sample)

            if self.obstacle_free(x_nearest, x_new):
                # Compute cost and find the optimal parent
                c_min = self._compute_cost_to_reach(x_nearest, x_new)
                x_min, c_min = self._select_parent(c_min, x_nearest, x_new, self.find_k_nearest(x_new, self.num_nearest_neighbors))

                # Add the new node to the tree
                self._add_node_to_tree(x_new, x_min, c_min)

                # Rewire the tree with the new node
                self._rewire_tree(x_new, x_min, self.find_k_nearest(x_new, self.num_nearest_neighbors))

                # Check if the new node reaches the goal
                if np.linalg.norm(np.array(x_new) - np.array(self.goal)) <= self.delta_distance:
                    if self.obstacle_free(x_new, self.goal):
                        # Compute cost for reaching the goal and update best path if necessary
                        c_goal = self.cost[x_new] + np.linalg.norm(np.array(self.goal) - np.array(x_new))
                        if self.goal not in self.cost or c_goal < self.cost[self.goal]:
                            self._add_node_to_tree(self.goal, x_new, c_goal)
                            self.best_path_length = c_goal  # Set best path length for informed sampling

        # Reconstruct the path if the goal was reached
        return self.reconstruct_path(self.goal) if self.goal in self.parent else None

    def _calculate_path_length(self, path):
        """Calculate the total length of a given path."""
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        return length