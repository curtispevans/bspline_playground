import numpy as np
from scipy.integrate import solve_ivp

def evolve_trajectory(C, D_matrix, basis_func_d1, v_d_func, M, tau_0=0.0, t_span=(0.0, 30.0), dt=0.01):
    """
    Evolves tau(t) under the arc-length speed profile constraint over tau in [0, M].
    
    C: (3, M+d) matrix of control points
    D_matrix: (M+d, M+d-1) derivative matrix operator
    basis_func_d1: callable evaluating b_M^{d-1}(tau) returning vector of length M+d-1
    v_d_func: callable returning desired scalar speed v_d(t)
    M: upper bound of the knot domain
    """
    def dtau_dt(t, tau):
        # 1. Clip tau to [0, M] domain
        tau_val = np.clip(tau[0], 0.0, M)
        
        # 2. Compute spatial derivative p'(tau) = C @ D @ b_d1(tau)
        b_d1 = basis_func_d1(tau_val)        # (M+d-1,)
        p_prime = C @ (D_matrix @ b_d1)      # (3,)
        
        speed_geom = np.linalg.norm(p_prime)
        v_d = v_d_func(t)
        
        return [v_d / (speed_geom + 1e-8)]

    # 3. Stop integration when tau reaches M
    def reached_end(t, tau):
        return tau[0] - M
    reached_end.terminal = True
    reached_end.direction = 1

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        dtau_dt, 
        t_span, 
        [tau_0], 
        t_eval=t_eval, 
        events=reached_end, 
        rtol=1e-6, 
        atol=1e-8
    )
    
    return sol.t, sol.y[0]  # Returns time grid and corresponding tau values