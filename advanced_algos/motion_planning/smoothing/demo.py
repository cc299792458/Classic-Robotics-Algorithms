import numpy as np

def univariate_time_optimal_interpolants(x1, x2, v1, v2, vmax, amax):
    """
    Compute the time-optimal trajectory execution time for univariate motion.

    Input:
    - x1, x2: Initial and final positions.
    - v1, v2: Initial and final velocities.
    - vmax: Maximum velocity.
    - amax: Maximum acceleration.

    Return:
    - T: Minimal execution time for valid motion primitive combinations, or None if no valid combination exists.
    """
    def solve_quadratic(a, b, c):
        """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return []
        sqrt_discriminant = np.sqrt(discriminant)
        return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]
    
    # Class P^+P^-
    def compute_p_plus_p_minus():
        coefficients = [
            amax,
            2 * v1,
            (v1**2 - v2**2) / (2 * amax) + x1 - x2
        ]
        solutions = solve_quadratic(*coefficients)
        valid_t = [t for t in solutions if max((v2 - v1) / amax, 0) <= t <= (vmax - v1) / amax]
        if not valid_t:
            return None
        t_p = valid_t[0]
        
        return np.array(2 * t_p + (v1 - v2) / amax)

    # Class P^-P^+
    def compute_p_minus_p_plus():
        coefficients = [
            amax,
            -2 * v1,
            (v1**2 - v2**2) / (2 * amax) + x2 - x1
        ]
        solutions = solve_quadratic(*coefficients)
        valid_t = [t for t in solutions if max((v1 - v2) / amax, 0) <= t <= (vmax + v1) / amax]
        if not valid_t:
            return None
        t_p = valid_t[0]
        
        return np.array(2 * t_p + (v2 - v1) / amax)

    # Class P^+L^+P^-
    def compute_p_plus_l_plus_p_minus():
        t_p1 = (vmax - v1) / amax
        t_p2 = (vmax - v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) + (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return np.array(t_p1 + t_l + t_p2)

    # Class P^-L^+P^+
    def compute_p_minus_l_plus_p_plus():
        t_p1 = (vmax + v1) / amax
        t_p2 = (vmax + v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) - (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return np.array(t_p1 + t_l + t_p2)

    # Evaluate all four classes in the specified order
    t_p_plus_p_minus = compute_p_plus_p_minus()
    t_p_minus_p_plus = compute_p_minus_p_plus()
    t_p_plus_l_plus_p_minus = compute_p_plus_l_plus_p_minus()
    t_p_minus_l_plus_p_plus = compute_p_minus_l_plus_p_plus()

    # Collect valid times and return the minimal one
    times = [t for t in [t_p_plus_p_minus, t_p_minus_p_plus, t_p_plus_l_plus_p_minus, t_p_minus_l_plus_p_plus] if t is not None]
    return np.min(times) if times else None

if __name__ == '__main__':
    # Example usage
    x1, x2 = 0, -0.5
    v1, v2 = 1, 1
    vmax, amax = 1, 1
    T = univariate_time_optimal_interpolants(x1, x2, v1, v2, vmax, amax)
    print(f"Minimal execution time: {T}")