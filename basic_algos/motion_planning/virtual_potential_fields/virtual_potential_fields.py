import numpy as np

class VirtualPotentialField:
    def __init__(self, goal, obstacles, obstacle_radii, k_att=0.5, k_rep=50.0, d_safe=1.5, step_size=0.05, max_repulsive_force=100.0):
        """
        Initialize the VPF parameters and environment.
        
        Parameters:
        - goal: The target position (np.array).
        - obstacles: List of obstacle positions (list of np.array).
        - obstacle_radii: List of radii defining the size of each obstacle.
        - k_att: Coefficient for attractive force.
        - k_rep: Coefficient for repulsive force.
        - d_safe: Safe distance for repulsive force activation.
        - step_size: Step size for each iteration.
        - max_repulsive_force: Maximum limit for the repulsive force.
        """
        self.goal = goal
        self.obstacles = obstacles
        self.obstacle_radii = obstacle_radii
        self.k_att = k_att
        self.k_rep = k_rep
        self.d_safe = d_safe
        self.step_size = step_size
        self.max_repulsive_force = max_repulsive_force
        self.positions = []

    def attractive_force(self, position):
        """
        Calculate the attractive force towards the goal.
        
        Parameters:
        - position: Current position of the robot (np.array).
        
        Returns:
        - Attractive force vector (np.array).
        """
        return self.k_att * (self.goal - position)

    def repulsive_force(self, position):
        """
        Calculate the total repulsive force from all obstacles, considering obstacle area.
        
        Parameters:
        - position: Current position of the robot (np.array).
        
        Returns:
        - Total repulsive force vector (np.array) with an upper limit.
        """
        force_total = np.array([0.0, 0.0])
        for obs, radius in zip(self.obstacles, self.obstacle_radii):
            distance = np.linalg.norm(position - obs) - radius  # Effective distance from the obstacle edge
            if distance < self.d_safe:
                # If within the obstacle's radius, treat repulsive force as infinite (large value)
                if distance <= 0:
                    distance = 1e-9  # Avoid division by zero, simulate very high repulsive force
                
                # Calculate raw repulsive force
                repulsive_force = self.k_rep * (1.0 / distance - 1.0 / self.d_safe) * (position - obs) / (distance ** 2)
                
                # Limit the repulsive force to the maximum threshold
                if np.linalg.norm(repulsive_force) > self.max_repulsive_force:
                    repulsive_force = repulsive_force / np.linalg.norm(repulsive_force) * self.max_repulsive_force
                
                force_total += repulsive_force
        return force_total

    def plan_path(self, start, max_iterations=500):
        """
        Plan the path using the Virtual Potential Field method.
        
        Parameters:
        - start: Starting position of the robot (np.array).
        - max_iterations: Maximum number of iterations to avoid infinite loop.
        
        Returns:
        - Path as a list of positions.
        """
        position = start.copy()  # Make a copy to avoid modifying the original start position
        self.positions.append(position.copy())

        for _ in range(max_iterations):
            # Calculate forces
            f_att = self.attractive_force(position)
            f_rep = self.repulsive_force(position)

            # Calculate total force and update position
            total_force = f_att + f_rep
            position += self.step_size * total_force
            self.positions.append(position.copy())

            # Check if goal is reached
            if np.linalg.norm(position - self.goal) < 0.1:
                print("Goal reached!")
                break

        return self.positions
