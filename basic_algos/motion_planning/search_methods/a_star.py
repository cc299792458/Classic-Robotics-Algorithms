import heapq

class AStarMotionPlanner:
    def __init__(self, start, goal, neighbors, cost, heuristic, constraints=None):
        self.start = start
        self.goal = goal
        self.neighbors = neighbors  # Function to get neighboring states
        self.cost = cost  # Function to calculate cost between states
        self.heuristic = heuristic  # Heuristic function for A*
        self.constraints = constraints  # Any additional constraints (e.g., collision checks)
        self.OPEN = []
        self.past_cost = {start: 0}
        self.parent = {start: None}
    
    def search(self):
        heapq.heappush(self.OPEN, (0, self.start))
        
        while self.OPEN:
            current_cost, current = heapq.heappop(self.OPEN)
            
            # Check if the current state meets the goal criteria
            if self.is_goal(current):
                return self.reconstruct_path(current)
            
            # Explore each neighboring state
            for nbr in self.neighbors(current):
                tentative_past_cost = self.past_cost[current] + self.cost(current, nbr)
                
                if self.is_valid(nbr) and (nbr not in self.past_cost or tentative_past_cost < self.past_cost[nbr]):
                    self.past_cost[nbr] = tentative_past_cost
                    self.parent[nbr] = current
                    estimated_total_cost = tentative_past_cost + self.heuristic(nbr, self.goal)
                    heapq.heappush(self.OPEN, (estimated_total_cost, nbr))
        
        return None  # If no path found, return failure
    
    def is_goal(self, current):
        # Check if the current state satisfies the goal condition (e.g., position tolerance)
        return current == self.goal
    
    def is_valid(self, state):
        # Check if a state is valid under any constraints (e.g., collision checks, joint limits)
        if self.constraints:
            return self.constraints(state)
        return True
    
    def reconstruct_path(self, current):
        path = []
        while current:
            path.append(current)
            current = self.parent[current]
        return path[::-1]  # Return the reversed path