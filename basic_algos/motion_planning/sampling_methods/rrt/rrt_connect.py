import numpy as np

from basic_algos.motion_planning.sampling_methods.rrt.rrt import RRT

class RRTConnect:
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacle_free = obstacle_free  # Function to check if the path between two points is collision-free
        self.max_iters = max_iters
        self.delta_distance = delta_distance  # Maximum extension distance
        self.sampling_range = sampling_range  # Sampling range as ((x_min, x_max), (y_min, y_max))
        self.tree = [self.start]  # Initialize the start tree with the start node
        self.parent = {self.start: None}  # Parent dictionary for the start tree
        self.goal_tree = [self.goal]  # Initialize the goal tree with the goal node
        self.goal_parent = {self.goal: None}  # Parent dictionary for the goal tree
        self.all_edges = []  # List to store all edges for visualization
        self.num_nodes = 2  # Initialize number of nodes with start and goal nodes

    def sample(self):
        """Randomly sample a point within the sampling range."""
        x_min, x_max = self.sampling_range[0]
        y_min, y_max = self.sampling_range[1]
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return (x, y)

    def nearest(self, tree, point):
        """Find the nearest node in the tree to the given point."""
        distances = [np.linalg.norm(np.array(node) - np.array(point)) for node in tree]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]

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

    def extend(self, tree, parent, x_nearest, x_target):
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
            if x_new == x_target:
                return 'Reached', x_new
            else:
                return 'Advanced', x_new
        return 'Trapped', x_nearest

    def connect(self, tree, parent, x_target):
        """
        Keep extending the tree towards x_target until it cannot advance.
        Returns a status ('Reached', 'Trapped') and the last node.
        """
        status = 'Advanced'
        x_nearest = self.nearest(tree, x_target)
        while True:
            status, x_new = self.extend(tree, parent, x_nearest, x_target)
            if status == 'Trapped':
                return 'Trapped', x_nearest
            if status == 'Reached':
                return 'Reached', x_new
            x_nearest = x_new

    def plan(self):
        """Run the RRT-Connect algorithm to find a path from start to goal."""
        for i in range(self.max_iters):
            x_sample = self.sample()

            # Extend the start tree towards the sampled point
            status, x_new_start = self.extend(
                self.tree, self.parent,
                self.nearest(self.tree, x_sample), x_sample
            )
            if status != 'Trapped':
                # Try to connect the goal tree to the new node in the start tree
                status_connect, x_new_goal = self.connect(
                    self.goal_tree, self.goal_parent, x_new_start
                )
                # Check if the trees have connected
                if status_connect == 'Reached':
                    return self.reconstruct_bidirectional_path(x_new_start, x_new_goal)

            # Swap the roles of the trees for balanced growth
            self.tree, self.goal_tree = self.goal_tree, self.tree
            self.parent, self.goal_parent = self.goal_parent, self.parent

        return None  # Return None if no path is found within max_iters

    def reconstruct_bidirectional_path(self, x_start_connect, x_goal_connect):
        """Reconstruct the path from start to goal using both trees."""
        # Path from start to the connection point in start tree
        path_start = []
        node = x_start_connect
        while node is not None:
            path_start.append(node)
            node = self.parent.get(node)
        path_start.reverse()

        # Path from goal to the connection point in goal tree
        path_goal = []
        node = x_goal_connect
        while node is not None:
            path_goal.append(node)
            node = self.goal_parent.get(node)

        # Combine the paths
        path = path_start + path_goal
        return path