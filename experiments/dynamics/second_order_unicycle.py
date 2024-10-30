import numpy as np
import matplotlib.pyplot as plt

from utils.dynamic_utils import SecondOrderUnicycle

# Simulate different motion paths: straight line, circular motion, and accelerating
def simulate_unicycle_motion(unicycle, initial_state, control_input_func, steps, dt, method):
    """
    Simulate the unicycle's motion over a given number of steps.
    
    Params:
    - unicycle (SecondOrderUnicycle): The dynamics model instance.
    - initial_state (np.array): Initial state of the system [x, y, v, theta].
    - control_input_func (function): Function to get control input at each step.
    - steps (int): Number of steps for the simulation.
    - dt (float): Time step.
    - method (str): Numerical method for integration ('euler' or 'rk4').

    Returns:
    - (list, list): x and y positions of the unicycle over time.
    """
    x_positions = []
    y_positions = []
    
    # Start simulation from the initial state
    state = initial_state
    
    for i in range(steps):
        x_positions.append(state[0])
        y_positions.append(state[1])
        # Get control input from the function
        control_input = control_input_func(i, steps)
        # Update the state using the dynamics model
        state = unicycle.step(state, control_input, dt, method)
    
    return x_positions, y_positions

# Create the unicycle dynamics instance
unicycle = SecondOrderUnicycle()

# Define time step
dt = 0.1

# Simulate straight-line motion
def control_input_straight_line(step, total_steps):
    # Constant velocity and no rotation (for straight-line motion)
    return np.array([0.0, 0.0])

initial_state_line = np.array([0.0, 0.0, 1.0, 0.0])  # [x, y, v, theta]
steps_line = 100  # Number of simulation steps
x_line, y_line = simulate_unicycle_motion(unicycle, initial_state_line, control_input_straight_line, steps_line, dt, method='euler')

# Simulate circular motion
def control_input_circle(step, total_steps):
    # Constant velocity and angular velocity for circular motion
    v_circle = 1.0
    r_circle = 0.25
    omega_circle = v_circle / r_circle
    return np.array([0.0, omega_circle])

initial_state_circle = np.array([0.0, -0.25, 1.0, np.pi / 2])  # Starting at (0, -0.25), facing up
steps_circle = int(2 * np.ceil(np.pi * 0.25 / (1.0 * dt))) + 1  # Number of steps to complete a circle
x_circle, y_circle = simulate_unicycle_motion(unicycle, initial_state_circle, control_input_circle, steps_circle, dt, method='euler')

# Simulate complex accelerating and decelerating motion (extended time for multiple spirals)
def complex_accel_control(step, total_steps):
    """
    Function to generate control input where both velocity and angular velocity
    accelerate for half the steps, then decelerate.
    """
    mid_point = total_steps // 2
    if step < mid_point:
        # Accelerating phase
        a_v = 0.05  # Linear acceleration
        omega = 0.1  # Constant angular velocity during acceleration
    else:
        # Decelerating phase
        a_v = -0.05  # Linear deceleration
        omega = -0.1  # Constant angular velocity during deceleration
    return np.array([a_v, omega])

initial_state_complex = np.array([0.0, 0.0, 0.0, 0.0])  # Starting from rest
steps_complex = 5000  # Extended number of steps for longer spiral motion
x_complex, y_complex = simulate_unicycle_motion(unicycle, initial_state_complex, complex_accel_control, steps_complex, dt, method='euler')

# Simulate random control input motion
def random_control(step, total_steps):
    """
    Generate random control inputs for each step.
    """
    a_v = np.random.uniform(-0.1, 0.1)  # Random linear acceleration
    omega = np.random.uniform(-0.5, 0.5)  # Random angular velocity
    return np.array([a_v, omega])

initial_state_random = np.array([0.0, 0.0, 0.5, 0.0])  # Start with initial speed
steps_random = 3000  # Number of steps for random motion
x_random, y_random = simulate_unicycle_motion(unicycle, initial_state_random, random_control, steps_random, dt, method='euler')

# Plot the results
plt.figure(figsize=(10, 10))

# Plot straight-line motion
plt.subplot(2, 2, 1)
plt.plot(x_line, y_line, label="Straight Line")
plt.title("Straight Line Motion")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.grid(True)

# Plot circular motion
plt.subplot(2, 2, 2)
plt.plot(x_circle, y_circle, label="Circular Path", color='orange')
plt.title("Circular Motion")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal scaling for x and y axes

# Plot complex accelerating and decelerating motion
plt.subplot(2, 2, 3)
plt.plot(x_complex, y_complex, label="Complex Accel-Decel Motion", color='purple')
plt.title("Complex Accelerating and Decelerating Motion (Extended)")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.grid(True)

# Plot random control motion
plt.subplot(2, 2, 4)
plt.plot(x_random, y_random, label="Random Motion", color='red')
plt.title("Random Control Motion")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.grid(True)

plt.tight_layout()
plt.show()
