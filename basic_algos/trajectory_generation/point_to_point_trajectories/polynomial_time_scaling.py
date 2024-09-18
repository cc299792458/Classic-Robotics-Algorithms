import numpy as np

def third_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, max_vel=None, max_acc=None, duration=None):
    """
    Generates a third-order polynomial time scaling for a given start and end position, considering maximum velocity
    and acceleration constraints.
    
    Args:
        start_pos (float): Initial position.
        end_pos (float): Final position.
        start_vel (float): Initial velocity.
        end_vel (float): Final velocity.
        max_vel (float, optional): Maximum allowable velocity.
        max_acc (float, optional): Maximum allowable acceleration.
        duration (float, optional): Desired time duration of the motion.
        
    Returns:
        tuple: (coefficients, duration)
            coefficients: (A, B, C, D) Coefficients of the third-order polynomial.
            duration: Calculated time duration of the motion.
    """
    # Calculate the total position change
    delta_pos = np.abs(end_pos - start_pos)
    
    # If max_vel or max_acc is provided, calculate the duration
    if max_vel is not None or max_acc is not None:
        # Calculate duration based on maximum velocity if provided
        duration_vel = 3 * delta_pos / (2 * max_vel) if max_vel is not None else 0.0
        
        # Calculate duration based on maximum acceleration if provided
        duration_acc = np.sqrt((6 * delta_pos) / max_acc) if max_acc is not None else 0.0
        
        # Choose the larger duration to ensure both constraints are satisfied
        duration = max(duration_vel, duration_acc, duration) if duration is not None else max(duration_vel, duration_acc)

    # Calculate coefficients using the precomputed formulas
    A = (2 * (start_pos - end_pos) + (start_vel + end_vel) * duration) / (duration ** 3)
    B = (3 * (end_pos - start_pos) - (2 * start_vel + end_vel) * duration) / (duration ** 2)
    C = start_vel
    D = start_pos

    return (A, B, C, D), duration

def fifth_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, start_acc=0, end_acc=0, max_vel=None, max_acc=None, duration=None):
    """
    Generates a fifth-order polynomial time scaling for a given start and end position, considering maximum velocity
    and acceleration constraints.
    
    Args:
        start_pos (float): Initial position.
        end_pos (float): Final position.
        start_vel (float): Initial velocity.
        end_vel (float): Final velocity.
        start_acc (float): Initial acceleration.
        end_acc (float): Final acceleration.
        max_vel (float, optional): Maximum allowable velocity.
        max_acc (float, optional): Maximum allowable acceleration.
        duration (float, optional): Desired time duration of the motion.
        
    Returns:
        tuple: (coefficients, duration)
            coefficients: (A, B, C, D, E, F) Coefficients of the fifth-order polynomial.
            duration: Calculated time duration of the motion.
    """
    # Calculate the total position change
    delta_pos = np.abs(end_pos - start_pos)

    # If max_vel or max_acc is provided, calculate the duration
    if max_vel is not None or max_acc is not None:
        # Estimate the duration needed to reach max velocity and acceleration
        # Peak velocity for a fifth-order polynomial is v_max = 15/8 * (delta_pos / duration)
        duration_vel = 15 * delta_pos / (8 * max_vel) if max_vel is not None else 0.0

        # Peak acceleration for a fifth-order polynomial is a_max = 10 * delta_pos / (T^2 * sqrt(3))
        duration_acc = np.sqrt(10 * delta_pos / (max_acc * np.sqrt(3))) if max_acc is not None else 0.0

        # Choose the larger duration to ensure both constraints are satisfied
        duration = max(duration_vel, duration_acc, duration) if duration is not None else max(duration_vel, duration_acc)

    # Calculate coefficients using the precomputed formulas
    A = (6 * (start_pos - end_pos) + 3 * (start_vel + end_vel) * duration + (end_acc - start_acc) * duration**2) / (2 * duration**5)
    B = (15 * (end_pos - start_pos) + (8 * start_vel + 7 * end_vel) * duration + (3 * start_acc - 2 * end_acc) * duration**2) / (2 * duration**4)
    C = (10 * (start_pos - end_pos) + (6 * end_vel + 4 * start_vel) * duration + (end_acc - start_acc) * duration**2) / (2 * duration**3)
    D = start_acc / 2
    E = start_vel
    F = start_pos

    return (A, B, C, D, E, F), duration

# Example usage
if __name__ == '__main__':
    start_pos = 0
    end_pos = 10
    duration = 5
    print(f"Desired time duration is {duration}")
    
    print("----- Third Order Polynomial -----")
    coefficients, duration = third_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, max_vel=1.0, max_acc=2.0)
    print("Coefficients (A, B, C, D):", coefficients)
    print(f"Duration is {duration}")
    
    print("----- Fifth Order Polynomial -----")
    coefficients, duration = fifth_order_polynomial_time_scaling(start_pos, end_pos, start_vel=0, end_vel=0, start_acc=0, end_acc=0, max_vel=1.0, max_acc=2.0)
    print("Coefficients (A, B, C, D, E, F):", coefficients)
    print(f"Duration is {duration}")
