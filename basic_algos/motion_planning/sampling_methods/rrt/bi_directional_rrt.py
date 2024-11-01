from basic_algos.motion_planning.sampling_methods.rrt.rrt import RRT

class BiDirectionalRRT(RRT):
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range):
        super().__init__(start, goal, obstacle_free, max_iters, delta_distance, sampling_range)
        self.goal_tree = [self.goal]  # Initialize the goal tree with the goal node
        self.goal_parent = {self.goal: None}  # Parent dictionary for the goal tree
        self.num_nodes = 2  # Start and goal nodes

    import numpy as np
from basic_algos.motion_planning.sampling_methods.rrt.rrt import RRT

class BiDirectionalRRT(RRT):
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range):
        super().__init__(start, goal, obstacle_free, max_iters, delta_distance, sampling_range)
        self.goal_tree = [self.goal]  # Initialize the goal tree with the goal node
        self.goal_parent = {self.goal: None}  # Parent dictionary for the goal tree
        self.num_nodes = 2  # Start and goal nodes

    def plan(self):
        """Run the Bi-Directional RRT algorithm to find a path from start to goal."""
        for _ in range(self.max_iters):
            x_sample = self.sample()

            # Extend the start tree towards the sampled point
            x_nearest_start = self.nearest(self.tree, x_sample)
            status_start, x_new_start = self.extend(self.tree, self.parent, x_nearest_start, x_sample)

            if status_start != 'Trapped':
                # Try to connect the goal tree to the new node in the start tree
                x_nearest_goal = self.nearest(self.goal_tree, x_new_start)
                status_connect, x_new_goal = self.extend(self.goal_tree, self.goal_parent, x_nearest_goal, x_new_start)

                # Check if the trees have connected
                if status_connect == 'Reached':
                    return self.reconstruct_path(x_new_start, x_new_goal)

            # Swap the roles of the trees for balanced growth
            self.tree, self.goal_tree = self.goal_tree, self.tree
            self.parent, self.goal_parent = self.goal_parent, self.parent

        return None  # Return None if no path is found within max_iters

    def reconstruct_path(self, x_start_connect, x_goal_connect):
        """Reconstruct the path from start to goal using both trees."""
        # Path from start to the connection point in the start tree
        path_start = []
        node = x_start_connect
        while node is not None:
            path_start.append(node)
            node = self.parent.get(node)
        path_start.reverse()

        # Path from goal to the connection point in the goal tree
        path_goal = []
        node = x_goal_connect
        while node is not None:
            path_goal.append(node)
            node = self.goal_parent.get(node)

        # Combine the paths
        path = path_start + path_goal
        return path

