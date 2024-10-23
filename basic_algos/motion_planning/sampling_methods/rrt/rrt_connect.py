import numpy as np

from basic_algos.motion_planning.sampling_methods.rrt.rrt import RRT

class RRTConnect(RRT):
    def __init__(self, start, goal, obstacle_free, max_iters, delta_distance, sampling_range):
        super().__init__(start, goal, obstacle_free, max_iters, delta_distance, sampling_range)
        self.goal_tree = [self.goal]  # Initialize goal tree
        self.goal_parent = {self.goal: None}  # Parent dictionary for the goal tree

    def extend(self, tree, parent, x_nearest, x_target):
        """Extend the tree towards the target point."""
        x_new = self.steer(x_nearest, x_target)
        if self.obstacle_free(x_nearest, x_new):
            tree.append(x_new)
            parent[x_new] = x_nearest
            if np.linalg.norm(np.array(x_new) - np.array(x_target)) < self.delta_distance:
                return 'REACHED', x_new
            else:
                return 'ADVANCED', x_new
        return 'TRAPPED', None

    def connect(self, tree, parent, x_target):
        """Try to connect the tree to the target point."""
        status = 'ADVANCED'
        x_nearest = self.nearest(x_target)
        while status == 'ADVANCED':
            status, x_new = self.extend(tree, parent, x_nearest, x_target)
            if status != 'TRAPPED':
                x_nearest = x_new
        return status, x_new

    def plan(self):
        """Run the RRT-Connect algorithm to find a path from start to goal."""
        for i in range(self.max_iters):
            # Sample a random point
            x_sample = self.sample()

            # Extend the start tree towards the sample
            status, x_new_start = self.extend(
                self.tree, self.parent,
                self.nearest(x_sample), x_sample
            )

            if status != 'TRAPPED':
                # Try to connect the goal tree to the new node in the start tree
                status_connect, x_new_goal = self.connect(
                    self.goal_tree, self.goal_parent, x_new_start
                )

                # Check if the trees have connected
                if status_connect == 'REACHED':
                    return self.reconstruct_bidirectional_path(x_new_start, x_new_goal)

            # Swap the roles of the trees
            self.tree, self.goal_tree = self.goal_tree, self.tree
            self.parent, self.goal_parent = self.goal_parent, self.parent

        return None  # Return failure if no path is found within max_iters

    def reconstruct_bidirectional_path(self, x_start_connect, x_goal_connect):
        """Reconstruct the path from start to goal using both trees."""
        # Path from start to x_start_connect
        path_start = []
        node = x_start_connect
        while node is not None:
            path_start.append(node)
            node = self.parent.get(node)
        path_start.reverse()

        # Path from x_goal_connect to goal
        path_goal = []
        node = x_goal_connect
        while node is not None:
            path_goal.append(node)
            node = self.goal_parent.get(node)

        # Combine paths
        path = path_start + path_goal
        return path
