import numpy as np

def minimum_acceleration_interpolants(x1, x2, v1, v2, vmax, T):
    """
    Compute the minimum-acceleration trajectory for fixed end time T.

    Input:
    - x1, x2: Initial and final positions.
    - v1, v2: Initial and final velocities.
    - vmax: Maximum velocity.
    - T: Fixed end time.

    Return:
    - a_min: Minimal acceleration for valid motion primitive combinations, or None if no valid combination exists.
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
            T**2,
            2 * T * (v1 + v2) + 4 * (x1 - x2),
            -(v2 - v1)**2
        ]
        solutions = solve_quadratic(*coefficients)
        valid_a = []
        for a in solutions:
            if a <= 0:
                continue
            t_s = 0.5 * (T + (v2 - v1) / a)
            if 0 < t_s < T and abs(v1 + a * t_s) <= vmax:
                valid_a.append(a)
        return min(valid_a) if valid_a else None

    # Class P^-P^+
    def compute_p_minus_p_plus():
        coefficients = [
            T**2,
            -2 * T * (v1 + v2) - 4 * (x1 - x2),
            -(v2 - v1)**2
        ]
        solutions = solve_quadratic(*coefficients)
        valid_a = []
        for a in solutions:
            if a <= 0:
                continue
            t_s = 0.5 * (T + (v1 - v2) / a)
            if 0 < t_s < T and abs(v1 - a * t_s) <= vmax:
                valid_a.append(a)
        return min(valid_a) if valid_a else None

    # Class P^+L^+P^-
    def compute_p_plus_l_plus_p_minus():
        a = (vmax**2 - vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax - (x2 - x1))
        if a <= 0:
            return None
        t_p1 = (vmax - v1) / a
        t_p2 = (vmax - v2) / a
        t_l = T - t_p1 - t_p2
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return a

    # Class P^-L^-P^+
    def compute_p_minus_l_minus_p_plus():
        a = (vmax**2 + vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax + (x2 - x1))
        if a <= 0:
            return None
        t_p1 = (vmax + v1) / a
        t_p2 = (vmax + v2) / a
        t_l = T - t_p1 - t_p2
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return a

    # Evaluate all four classes independently
    results = [
        compute_p_plus_p_minus(),  # P^+P^-
        compute_p_minus_p_plus(),  # P^-P^+
        compute_p_plus_l_plus_p_minus(),  # P^+L^+P^-
        compute_p_minus_l_minus_p_plus()  # P^-L^-P^+
    ]
    valid_results = [a for a in results if a is not None]

    return min(valid_results) if valid_results else None

# Example usage
if __name__ == '__main__':
    x1, x2 = 0, -0.5
    v1, v2 = 1, 1
    vmax = 1
    T = 4.5

    a_min = minimum_acceleration_interpolants(x1, x2, v1, v2, vmax, T)
    print(f"Minimal acceleration: {a_min}")
