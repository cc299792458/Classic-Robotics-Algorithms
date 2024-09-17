import numpy as np

def third_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, duration=1.0):
    """
    Generates a third-order polynomial time scaling for a given start and end position.
    
    Args:
        start_pos (float): Initial position.
        end_pos (float): Final position.
        start_vel (float): Initial velocity.
        end_vel (float): Final velocity.
        duration (float): Time duration of the motion.
        
    Returns:
        A, B, C, D (tuple): Coefficients of the third-order polynomial.
    """
    # Define the system of equations for the third-order polynomial:
    # p(t) = A*t^3 + B*t^2 + C*t + D
    # v(t) = 3*A*t^2 + 2*B*t + C

    # Matrix of equations
    M = np.array([
        [0, 0, 0, 1],
        [duration**3, duration**2, duration, 1],
        [0, 0, 1, 0],
        [3 * duration**2, 2 * duration, 1, 0]
    ])

    # Results vector
    b = np.array([start_pos, end_pos, start_vel, end_vel])

    # Solve for coefficients
    coefficients = np.linalg.solve(M, b)

    return coefficients

def fifth_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, start_acc=0, end_acc=0, duration=1.0):
    """
    Generates a fifth-order polynomial time scaling for a given start and end position.
    
    Args:
        start_pos (float): Initial position.
        end_pos (float): Final position.
        start_vel (float): Initial velocity.
        end_vel (float): Final velocity.
        start_acc (float): Initial acceleration.
        end_acc (float): Final acceleration.
        duration (float): Time duration of the motion.
        
    Returns:
        tuple: Coefficients of the fifth-order polynomial.
    """
    # Define the system of equations for the fifth-order polynomial:
    # p(t) = A*t^5 + B*t^4 + C*t^3 + D*t^2 + E*t + F
    # v(t) = 5*A*t^4 + 4*B*t^3 + 3*C*t^2 + 2*D*t + E
    # a(t) = 20*A*t^3 + 12*B*t^2 + 6*C*t + 2*D

    # Matrix of equations
    M = np.array([
        [0, 0, 0, 0, 0, 1],
        [duration**5, duration**4, duration**3, duration**2, duration, 1],
        [0, 0, 0, 0, 1, 0],
        [5*duration**4, 4*duration**3, 3*duration**2, 2*duration, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [20*duration**3, 12*duration**2, 6*duration, 2, 0, 0]
    ])

    # Results vector
    b = np.array([start_pos, end_pos, start_vel, end_vel, start_acc, end_acc])

    # Solve for coefficients
    coefficients = np.linalg.solve(M, b)

    return coefficients

# Example usage
if __name__ == '__main__':
    start_pos = 0
    end_pos = 10
    duration = 5
    coefficients = third_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, duration=duration)
    print("Third Order Polynomial Coefficients (A, B, C, D):", coefficients)

    coefficients = fifth_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, start_acc=0, end_acc=0, duration=duration)
    print("Fifth Order Polynomial Coefficients (A, B, C, D, E, F):", coefficients)