import numpy as np
import matplotlib.pyplot as plt

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
    - selected_primitive: Name of the selected motion primitive.
    - params: Parameters required to reconstruct the trajectory.
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
        T = 2 * t_p + (v1 - v2) / amax
        params = {'t_p': t_p}
        return T, 'P^+P^-', params

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
        T = 2 * t_p + (v2 - v1) / amax
        params = {'t_p': t_p}
        return T, 'P^-P^+', params

    # Class P^+L^+P^-
    def compute_p_plus_l_plus_p_minus():
        t_p1 = (vmax - v1) / amax
        t_p2 = (vmax - v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) + (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        T = t_p1 + t_l + t_p2
        params = {'t_p1': t_p1, 't_l': t_l, 't_p2': t_p2}
        return T, 'P^+L^+P^-', params

    # Class P^-L^+P^+
    def compute_p_minus_l_plus_p_plus():
        t_p1 = (vmax + v1) / amax
        t_p2 = (vmax + v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) - (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        T = t_p1 + t_l + t_p2
        params = {'t_p1': t_p1, 't_l': t_l, 't_p2': t_p2}
        return T, 'P^-L^+P^+', params

    # Evaluate all four classes in the specified order
    results = []
    for compute_func in [compute_p_plus_p_minus, compute_p_minus_p_plus, compute_p_plus_l_plus_p_minus, compute_p_minus_l_plus_p_plus]:
        result = compute_func()
        if result is not None:
            results.append(result)
    
    if not results:
        return None, None, None
    
    # Find the minimal execution time
    T_values = [result[0] for result in results]
    min_index = np.argmin(T_values)
    T, selected_primitive, params = results[min_index]
    return T, selected_primitive, params

if __name__ == '__main__':
    x1, x2 = 0, 1
    v1, v2 = 0.1, 0
    vmax, amax = 1, 1
    T, selected_primitive, params = univariate_time_optimal_interpolants(x1, x2, v1, v2, vmax, amax)
    if T is None:
        print("No valid trajectory found.")
        exit()
    print(f"Minimal execution time: {T}")
    print(f"Selected motion primitive: {selected_primitive}")
    
    # Generate time steps
    num_points = 1000
    t_array = np.linspace(0, T, num_points)
    x_array = np.zeros_like(t_array)
    v_array = np.zeros_like(t_array)
    
    # Compute x(t) and v(t) based on the selected primitive and parameters
    if selected_primitive == 'P^+P^-':
        t_p = params['t_p']
        t_q = T - t_p
        for i, t in enumerate(t_array):
            if t <= t_p:
                # Acceleration phase
                v_array[i] = v1 + amax * t
                x_array[i] = x1 + v1 * t + 0.5 * amax * t**2
            else:
                # Deceleration phase
                delta_t = t - t_p
                v_peak = v1 + amax * t_p
                v_array[i] = v_peak - amax * delta_t
                x_array[i] = x1 + v1 * t_p + 0.5 * amax * t_p**2 + v_peak * delta_t - 0.5 * amax * delta_t**2
    elif selected_primitive == 'P^-P^+':
        t_p = params['t_p']
        t_q = T - t_p
        for i, t in enumerate(t_array):
            if t <= t_p:
                # Deceleration phase
                v_array[i] = v1 - amax * t
                x_array[i] = x1 + v1 * t - 0.5 * amax * t**2
            else:
                # Acceleration phase
                delta_t = t - t_p
                v_valley = v1 - amax * t_p
                v_array[i] = v_valley + amax * delta_t
                x_array[i] = x1 + v1 * t_p - 0.5 * amax * t_p**2 + v_valley * delta_t + 0.5 * amax * delta_t**2
    elif selected_primitive == 'P^+L^+P^-':
        t_p1 = params['t_p1']
        t_l = params['t_l']
        t_p2 = params['t_p2']
        for i, t in enumerate(t_array):
            if t <= t_p1:
                # Acceleration phase
                v_array[i] = v1 + amax * t
                x_array[i] = x1 + v1 * t + 0.5 * amax * t**2
            elif t <= t_p1 + t_l:
                # Constant velocity phase
                v_array[i] = vmax
                delta_t = t - t_p1
                x_array[i] = x1 + v1 * t_p1 + 0.5 * amax * t_p1**2 + vmax * delta_t
            else:
                # Deceleration phase
                delta_t = t - t_p1 - t_l
                v_array[i] = vmax - amax * delta_t
                x_array[i] = x1 + v1 * t_p1 + 0.5 * amax * t_p1**2 + vmax * t_l + vmax * delta_t - 0.5 * amax * delta_t**2
    elif selected_primitive == 'P^-L^+P^+':
        t_p1 = params['t_p1']
        t_l = params['t_l']
        t_p2 = params['t_p2']
        for i, t in enumerate(t_array):
            if t <= t_p1:
                # Deceleration phase
                v_array[i] = v1 - amax * t
                x_array[i] = x1 + v1 * t - 0.5 * amax * t**2
            elif t <= t_p1 + t_l:
                # Constant negative velocity phase
                v_array[i] = -vmax
                delta_t = t - t_p1
                x_array[i] = x1 + v1 * t_p1 - 0.5 * amax * t_p1**2 + (-vmax) * delta_t
            else:
                # Acceleration phase
                delta_t = t - t_p1 - t_l
                v_array[i] = -vmax + amax * delta_t
                x_array[i] = x1 + v1 * t_p1 - 0.5 * amax * t_p1**2 + (-vmax) * t_l + (-vmax) * delta_t + 0.5 * amax * delta_t**2
    else:
        print("Unknown motion primitive.")
        exit()
    
    # Plot x(t) and v(t)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_array, x_array)
    plt.title('Position x(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_array, v_array)
    plt.title('Velocity v(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
