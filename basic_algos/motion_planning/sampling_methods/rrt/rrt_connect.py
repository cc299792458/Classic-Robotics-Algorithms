import numpy as np

from basic_algos.motion_planning.sampling_methods.rrt.rrt import RRT

class RRTConnect(RRT):
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range):
        super().__init__(start, goal, obstacle_free, max_iters, delta_distance, sampling_range)
        self.goal_tree = [self.goal]  # Initialize the goal tree with the goal node
        self.goal_parent = {self.goal: None}  # Parent dictionary for the goal tree
        self.num_nodes = 2  # Initialize number of nodes with start and goal nodes

    def connect(self, tree, parent, x_target):
        """
        Keep extending the tree towards x_target until it cannot advance.
        Returns a status ('Reached', 'Trapped') and the last node.
        """
        status = 'Advanced'
        x_nearest = self.nearest(tree, x_target)
        while status == 'Advanced':
            status, x_new = self.extend(tree, parent, x_nearest, x_target)
            if status == 'Trapped':
                return 'Trapped', x_nearest
            if status == 'Reached':
                return 'Reached', x_new
            x_nearest = x_new  # Continue extending from the new node
        return status, x_nearest

    def plan(self):
        """Run the RRT-Connect algorithm to find a path from start to goal."""
        for _ in range(self.max_iters):
            x_sample = self.sample()

            # Extend the start tree towards the sampled point
            x_nearest_start = self.nearest(self.tree, x_sample)
            status, x_new_start = self.extend(
                self.tree, self.parent,
                x_nearest_start, x_sample
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
