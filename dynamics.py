import jax.numpy as jnp

def get_external_current(t, I_type, I_val):
    """Generates external stimulus current dynamically at time t."""
    constant_current = I_val
    step_current = jnp.where((t >= 10.0) & (t <= 80.0), I_val, 0.0)
    sine_current = I_val * (1.0 + 0.5 * jnp.sin(0.2 * t))
    pulse_current = jnp.where(jnp.mod(t, 20.0) <= 5.0, I_val, 0.0)
    
    if I_type == 'step':
        return step_current
    elif I_type == 'sine':
        return sine_current
    elif I_type == 'pulse':
        return pulse_current
    else:
        return constant_current

def fhn_vector_field(t, y, args):
    """Computes the vector field for the FitzHugh-Nagumo ordinary differential equations."""
    v, w = y
    a, b, tau, I_type, I_val = args
    I_ext = get_external_current(t, I_type, I_val)
    dv_dt = v - (v**3) / 3.0 - w + I_ext
    dw_dt = (v + a - b * w) / tau
    return jnp.stack([dv_dt, dw_dt])

def run_hankel_dmd(v_data, H, r):
    """Computes Hankel Dynamic Mode Decomposition (Hankel-DMD) on time-series data."""
    v_data = jnp.asarray(v_data, dtype=jnp.float32)
    T = len(v_data)
    
    if H >= T:
        raise ValueError(f"Hankel delay embedding H ({H}) must be strictly less than time series length T ({T})")
        
    K = T - H + 1
    H_matrix = jnp.stack([v_data[i : i + K] for i in range(H)], axis=0)
    X = H_matrix[:, :-1]
    Y = H_matrix[:, 1:]
    
    U, s, V_T = jnp.linalg.svd(X, full_matrices=False)
    V = V_T.T
    
    r = min(r, U.shape[1])
    U_r = U[:, :r]
    V_r = V[:, :r]
    s_r = s[:r]
    
    Sigma_inv = jnp.diag(1.0 / s_r)
    A = U_r.T @ Y @ V_r @ Sigma_inv
    eigenvalues = jnp.linalg.eigvals(A)
    
    return A, eigenvalues, s, X, Y

def run_dmdc(v_data, u_data, H, r, p):
    """Computes Bilinear DMDc supporting single or multi-trajectory batches."""
    v_data = jnp.asarray(v_data, dtype=jnp.float32)
    u_data = jnp.asarray(u_data, dtype=jnp.float32)
    
    if v_data.ndim == 1:
        v_data = v_data[jnp.newaxis, :]
        u_data = u_data[jnp.newaxis, :]
        
    M, T = v_data.shape
    if H >= T:
        raise ValueError(f"Hankel delay H ({H}) must be strictly less than time series length T ({T})")
        
    K = T - H + 1
    
    X_list = []
    Y_list = []
    Omega_list = []
    Uc_list = []
    
    for m in range(M):
        v_traj = v_data[m]
        u_traj = u_data[m]
        
        H_state = jnp.stack([v_traj[i : i + K] for i in range(H)], axis=0)
        U_c = jnp.stack([u_traj[i + H - 1] for i in range(K - 1)], axis=0).reshape(1, -1)
        
        X_m = H_state[:, :-1]
        Y_m = H_state[:, 1:]
        X_kron_U = X_m * U_c
        Omega_m = jnp.concatenate([X_m, U_c, X_kron_U], axis=0)
        
        X_list.append(X_m)
        Y_list.append(Y_m)
        Omega_list.append(Omega_m)
        Uc_list.append(U_c)
        
    X_global = jnp.concatenate(X_list, axis=1)
    Y_global = jnp.concatenate(Y_list, axis=1)
    Omega_global = jnp.concatenate(Omega_list, axis=1)
    Uc_global = jnp.concatenate(Uc_list, axis=1)
    
    U_tilde, s_p, V_p_T = jnp.linalg.svd(Omega_global, full_matrices=False)
    V_p = V_p_T.T
    
    p = min(p, U_tilde.shape[1])
    U_p = U_tilde[:, :p]
    V_p = V_p[:, :p]
    s_p_r = s_p[:p]
    
    U_p1 = U_p[:H, :]
    U_p2 = U_p[H:H+1, :]
    U_p3 = U_p[H+1:, :]
    
    U_x, s_x, V_x_T = jnp.linalg.svd(X_global, full_matrices=False)
    r = min(r, U_x.shape[1])
    U_r = U_x[:, :r]
    
    Sigma_p_inv = jnp.diag(1.0 / s_p_r)
    A_tilde = U_r.T @ Y_global @ V_p @ Sigma_p_inv @ U_p1.T @ U_r
    B_tilde = U_r.T @ Y_global @ V_p @ Sigma_p_inv @ U_p2.T
    C_tilde = U_r.T @ Y_global @ V_p @ Sigma_p_inv @ U_p3.T @ U_r
    
    eigenvalues = jnp.linalg.eigvals(A_tilde)
    
    return A_tilde, B_tilde, C_tilde, eigenvalues, s_x, s_p, X_global, Y_global, Uc_global
