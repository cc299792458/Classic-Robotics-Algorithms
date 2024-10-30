import numpy as np

from scipy.interpolate import CubicSpline

class SS:
    """
    A single-query, single-directional, kinodynamic planning method with time-constrained spline goal connection.
    """

    def __init__(
        self,
        start_state,
        goal_state,
        state_space_bounds,
        control_space_bounds,
        dynamics,
        collision_checker,
        dt=0.1,
        control_duration=1.0,
        goal_tolerance=0.1,
        max_spline_time=1.0,  # Adjusted to 1.0
        bin_resolution=10,  # Default resolution for each dimension
    ):
        """
        Initialize the SS planner for single-directional search with time-constrained spline connection.

        Params:
        - start_state (np.array): The starting state for the planner.
        - goal_state (np.array): The goal state for the planner.
        - state_space_bounds (tuple): A tuple of (lower_bound, upper_bound) representing the state space range (continuous).
        - control_space_bounds (tuple): A tuple of (lower_bound, upper_bound) representing the control space range (continuous).
        - dynamics (Dynamics): An instance of a dynamics model that can simulate the system's evolution.
        - collision_checker (function): A function that checks for collisions between two states.
        - dt (float): Time step for the simulation of the system's evolution (discrete time approximation).
        - control_duration (float): The total duration for which the sampled control input is applied to evolve the system.
        - goal_tolerance (float): The tolerance distance within which we consider the goal reached.
        - max_spline_time (float): Maximum allowed time for the spline connection to the goal.
        - bin_resolution (int): Resolution for binning the state space in each dimension.
        """
        self.start_state = start_state
        self.goal_state = goal_state
        self.state_space_bounds = state_space_bounds
        self.control_space_bounds = control_space_bounds
        self.dynamics = dynamics
        self.collision_checker = collision_checker
        self.dt = dt
        self.control_duration = control_duration
        self.goal_tolerance = goal_tolerance
        self.max_spline_time = max_spline_time  # Adjusted to 1.0
        self.bin_resolution = bin_resolution
        
        # Tree structure: stores each node with its parent and control input
        self.tree = {tuple(start_state): (None, None)}  # Initial tree contains only start_state
        self.bins = self.initialize_bins()  # Initialize bins to store nodes
        self.solution_path = None
        self.is_solved = False

        # Add start node to the appropriate bin
        self.add_node_to_bin(start_state)

    def initialize_bins(self):
        """
        Initialize bins for dividing the state space. The number of bins in each dimension is determined
        by the dimensionality of the state space.

        Returns:
        - list of lists: A list of bins to store nodes.
        """
        state_dimension = len(self.start_state)
        # Create a multi-dimensional array (list of lists) to store bins based on state space dimensions
        bins = np.empty([self.bin_resolution] * state_dimension, dtype=object)
        for idx in np.ndindex(bins.shape):
            bins[idx] = []
        return bins

    def add_node_to_bin(self, state):
        """
        Add a node to the appropriate bin based on its state.

        Params:
        - state (np.array): The state of the node to add to the bin.

        Returns:
        - None
        """
        bin_index = self.get_bin_index(state)
        self.bins[bin_index].append(tuple(state))

    def get_bin_index(self, state):
        """
        Get the bin index for a given state.

        Params:
        - state (np.array): The state to find the bin index for.

        Returns:
        - tuple: The bin index corresponding to the state.
        """
        lower_bound_state, upper_bound_state = self.state_space_bounds
        # Normalize state to a range [0, 1] within the state space bounds
        normalized_state = (state - lower_bound_state) / (upper_bound_state - lower_bound_state)
        # Scale the normalized state to the number of bins
        bin_index = tuple((normalized_state * self.bin_resolution).astype(int).clip(0, self.bin_resolution - 1))
        return bin_index

    def get_random_node(self):
        """
        Return a random node from the tree by sampling a bin first, then sampling a node in the bin.

        Returns:
        - np.array: A random node from the tree.
        """
        # Step 1: Randomly sample a non-empty bin
        non_empty_bins = [bin_idx for bin_idx in np.ndindex(self.bins.shape) if len(self.bins[bin_idx]) > 0]
        random_bin_idx = non_empty_bins[np.random.randint(len(non_empty_bins))]

        # Step 2: Randomly sample a node from the selected bin
        random_node = self.bins[random_bin_idx][np.random.randint(len(self.bins[random_bin_idx]))]
        return np.array(random_node)

    def plan(self):
        """
        Perform the single-directional kinodynamic planning.
        """
        while not self.is_solved:
            # Expand the tree in the forward direction
            self.expand()

            # Try to connect the last node to the goal using spline connection
            if self.connect_to_goal(self.get_last_node()):
                self.construct_solution()
                return self.solution_path

        return self.solution_path

    def expand(self):
        """
        Expand the tree in the forward direction.

        Returns:
        - None: This function modifies the internal tree based on the expansion.
        """
        # Step 1: Select a random node from the tree
        current_node = self.get_random_node()

        # Step 2: Sample a random control input from the continuous control space
        lower_bound_control, upper_bound_control = self.control_space_bounds
        control_input = np.random.uniform(lower_bound_control, upper_bound_control)

        # Step 3: Simulate the system's evolution over the sampled duration control_duration
        new_state = np.copy(current_node)
        t = 0
        while t < self.control_duration:
            new_state = self.dynamics.step(new_state, control_input, self.dt)
            # Ensure the new state is within the state space bounds
            lower_bound_state, upper_bound_state = self.state_space_bounds
            new_state = np.clip(new_state, lower_bound_state, upper_bound_state)
            t += self.dt

        # Step 4: After control_duration, perform collision checking between the start and end states
        if not self.collision_checker(current_node, new_state):
            # Step 5: If no collision occurs, add the new state to the tree with its parent and control input
            self.tree[tuple(new_state)] = (tuple(current_node), control_input)
            self.add_node_to_bin(new_state)

    def connect_to_goal(self, state):
        """
        Try to connect the current state to the goal using a time-constrained cubic spline. If successful, return True.

        Params:
        - state (np.array): The current state of the system.

        Returns:
        - bool: True if the spline connection to the goal is collision-free, False otherwise.
        """
        # Generate a cubic spline connecting the current state to the goal over a time range [0, max_spline_time]
        spline_time_points = [0, self.max_spline_time]  # Start at 0, end at max_spline_time
        spline = CubicSpline(spline_time_points, np.vstack([state, self.goal_state]), axis=0)

        # Discretize the spline path based on time
        num_points = 20  # Number of points to sample along the spline
        time_samples = np.linspace(0, self.max_spline_time, num_points)

        for t in time_samples:
            interpolated_state = spline(t)
            if self.collision_checker(state, interpolated_state):
                return False  # Collision detected along the spline path

        # If no collision, we successfully connected to the goal
        self.tree[tuple(self.goal_state)] = (tuple(state), None)  # No control input needed for final connection
        return True

    def construct_solution(self):
        """
        Construct the solution path once the goal is reached.

        Returns:
        - None: Modifies self.solution_path with the states and controls leading to the goal.
        """
        # Trace back from goal to start, collecting states and controls
        path = []
        controls = []
        current_state = tuple(self.goal_state)
        
        while current_state is not None:
            parent_state, control_input = self.tree[current_state]
            path.append(np.array(current_state))  # Add current state to path
            if control_input is not None:
                controls.append(control_input)  # Add control input to controls
            current_state = parent_state
        
        path.reverse()  # Reverse the path to start from the initial state
        controls.reverse()
        self.solution_path = (path, controls)

    def get_last_node(self):
        """
        Return the last added node to the tree.

        Returns:
        - np.array: The last added node in the tree.
        """
        return np.array(list(self.tree.keys())[-1])
