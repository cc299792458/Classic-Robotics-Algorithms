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

class FirstOrderOmnidirectionalRobot(Dynamics):
    """
    First-order omnidirectional robot dynamics model allowing motion in any direction
    without changing its orientation.
    """

    def dynamics_equation(self, state, control):
        """
        Define the kinematic equation for a first-order omnidirectional robot.

        Params:
        - state (np.array): The current state [x, y, phi], where:
            - x and y are the position coordinates.
            - phi is the robot's orientation (heading angle).

        - control (np.array): The control input [v_x, v_y, omega], where:
            - v_x and v_y are the translational velocities in the robot's x and y directions.
            - omega is the rotational velocity around the z-axis.

        Returns:
        - np.array: The state derivatives [dx/dt, dy/dt, dphi/dt].
        """
        # Extract state variables
        x, y, phi = state

        # Extract control inputs
        v_x, v_y, omega = control

        # Dynamics equations for first-order omnidirectional robot
        dx_dt = v_x * np.cos(phi) - v_y * np.sin(phi)
        dy_dt = v_x * np.sin(phi) + v_y * np.cos(phi)
        dphi_dt = omega

        return np.array([dx_dt, dy_dt, dphi_dt])
    
class SecondOrderOmnidirectionalRobot(Dynamics):
    """
    Second-order omnidirectional robot dynamics model allowing motion in any direction
    with acceleration control.
    """

    def dynamics_equation(self, state, control):
        """
        Define the dynamics equation for a second-order omnidirectional robot.

        Params:
        - state (np.array): The current state [x, y, v_x, v_y, phi, omega], where:
            - x and y are the position coordinates.
            - v_x and v_y are the translational velocities in the robot's x and y directions.
            - phi is the robot's orientation (heading angle).
            - omega is the rotational velocity around the z-axis.

        - control (np.array): The control input [a_x, a_y, alpha], where:
            - a_x and a_y are the translational accelerations in the robot's x and y directions.
            - alpha is the rotational acceleration around the z-axis.

        Returns:
        - np.array: The state derivatives [dx/dt, dy/dt, dv_x/dt, dv_y/dt, dphi/dt, domega/dt].
        """
        # Extract state variables
        x, y, v_x, v_y, phi, omega = state

        # Extract control inputs
        a_x, a_y, alpha = control

        # Dynamics equations for second-order omnidirectional robot
        dx_dt = v_x
        dy_dt = v_y
        dv_x_dt = a_x
        dv_y_dt = a_y
        dphi_dt = omega
        domega_dt = alpha

        return np.array([dx_dt, dy_dt, dv_x_dt, dv_y_dt, dphi_dt, domega_dt])

class FirstOrderUnicycle(Dynamics):
    """
    First-order unicycle dynamics, inheriting from the base Dynamics class.
    Implements the dynamics_equation method for the first-order unicycle model.
    """

    def dynamics_equation(self, state, control):
        """
        Define the dynamics equation for a first-order unicycle.

        Params:
        - state (np.array): The current state [x, y, phi].
        - control (np.array): The control input [v, omega], where:
            - v is the linear velocity (speed along the direction of the vehicle).
            - omega is the angular velocity (rate of change of orientation).

        Returns:
        - np.array: The state derivatives [dx/dt, dy/dt, dphi/dt].
        """
        # Extract state variables
        x, y, phi = state

        # Extract control inputs
        v, omega = control

        # Dynamics equations for first-order unicycle
        dx_dt = v * np.cos(phi)     # Change in x (linear velocity in x direction)
        dy_dt = v * np.sin(phi)     # Change in y (linear velocity in y direction)
        dphi_dt = omega             # Change in orientation (angular velocity)

        # Return the state derivatives
        return np.array([dx_dt, dy_dt, dphi_dt])

class SecondOrderUnicycle(Dynamics):
    """
    Second-order unicycle dynamics, inheriting from the base Dynamics class.
    Implements the dynamics_equation method for the second-order unicycle model.
    """

    def dynamics_equation(self, state, control):
        """
        Define the dynamics equation for a second-order unicycle.

        Params:
        - state (np.array): The current state [x, y, v, phi].
        - control (np.array): The control input [a_v, omega], where:
            - a_v is the linear acceleration (change in velocity).
            - omega is the angular velocity (change in orientation).

        Returns:
        - np.array: The state derivatives [dx/dt, dy/dt, dv/dt, dphi/dt].
        """
        # Extract state variables
        x, y, v, phi = state

        # Extract control inputs
        a_v, omega = control

        # Dynamics equations for second-order unicycle
        dx_dt = v * np.cos(phi)     # Change in x (linear velocity in x direction)
        dy_dt = v * np.sin(phi)     # Change in y (linear velocity in y direction)
        dv_dt = a_v                 # Change in velocity (linear acceleration)
        dphi_dt = omega             # Change in orientation (angular velocity)

        # Return the state derivatives
        return np.array([dx_dt, dy_dt, dv_dt, dphi_dt])
