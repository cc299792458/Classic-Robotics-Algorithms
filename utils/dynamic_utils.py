import numpy as np

class Dynamics:
    """
    Base class for system dynamics.
    All specific dynamics models should inherit from this class and implement the step method.
    """

    def step(self, state, control, dt, method='euler'):
        """
        Simulate one time step of the system's dynamics using the specified integration method.

        Params:
        - state (np.array): The current state of the system.
        - control (np.array): The control input to the system.
        - dt (float): The time step for evolution.
        - method (str): The numerical integration method to use ('euler' or 'rk4').

        Returns:
        - np.array: The new state after applying the control for one time step dt.
        """
        if method == 'euler':
            return self.euler_step(state, control, dt)
        elif method == 'rk4':
            return self.rk4_step(state, control, dt)
        else:
            raise ValueError("Unsupported integration method.")

    def euler_step(self, state, control, dt):
        """
        Euler method to compute the next state.

        Params:
        - state (np.array): The current state of the system.
        - control (np.array): The control input to the system.
        - dt (float): The time step for evolution.

        Returns:
        - np.array: The new state after applying the control for one time step dt using Euler method.
        """
        return state + self.dynamics_equation(state, control) * dt

    def rk4_step(self, state, control, dt):
        """
        Runge-Kutta 4th order method to compute the next state.

        Params:
        - state (np.array): The current state of the system.
        - control (np.array): The control input to the system.
        - dt (float): The time step for evolution.

        Returns:
        - np.array: The new state after applying the control for one time step dt using RK4 method.
        """
        k1 = self.dynamics_equation(state, control)
        k2 = self.dynamics_equation(state + 0.5 * dt * k1, control)
        k3 = self.dynamics_equation(state + 0.5 * dt * k2, control)
        k4 = self.dynamics_equation(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def dynamics_equation(self, state, control):
        """
        Define the system's dynamics equation.
        This method should be implemented by subclasses to define specific dynamics.

        Params:
        - state (np.array): The current state of the system.
        - control (np.array): The control input to the system.

        Returns:
        - np.array: The state derivative (e.g., velocity or acceleration) based on the control input.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class SecondOrderUnicycle(Dynamics):
    """
    Second-order unicycle dynamics, inheriting from the base Dynamics class.
    Implements the dynamics_equation method for the second-order unicycle model.
    """

    def dynamics_equation(self, state, control):
        """
        Define the dynamics equation for a second-order unicycle.

        Params:
        - state (np.array): The current state [x, y, v, theta].
        - control (np.array): The control input [a_v, omega], where:
            - a_v is the linear acceleration (change in velocity).
            - omega is the angular velocity (change in orientation).

        Returns:
        - np.array: The state derivatives [dx/dt, dy/dt, dv/dt, dtheta/dt].
        """
        # Extract state variables
        x, y, v, theta = state

        # Extract control inputs
        a_v, omega = control

        # Dynamics equations for second-order unicycle
        dx_dt = v * np.cos(theta)        # Change in x (linear velocity in x direction)
        dy_dt = v * np.sin(theta)        # Change in y (linear velocity in y direction)
        dv_dt = a_v                      # Change in velocity (linear acceleration)
        dtheta_dt = omega                # Change in orientation (angular velocity)

        # Return the state derivatives
        return np.array([dx_dt, dy_dt, dv_dt, dtheta_dt])
