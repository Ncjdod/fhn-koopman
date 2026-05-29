"""
FitzHugh-Nagumo Neural Model Simulator and Parameter Estimator
--------------------------------------------------------------
A JAX-based implementation of the FitzHugh-Nagumo (FHN) model.
- Solves FHN dynamics using Diffrax adaptive ODE solvers (Tsit5).
- Performs gradient-based parameter estimation using Optax and JAX autodiff.
- Exports generated time-series data to CSV.
- Displays high-resolution scientific plots of time series and phase space with nullclines.

Repository location:
scientific-analysis-and-simulations/system-analysis/Organoid Intelligence--Dynamic Mode Decomposition/fitzhugh_nagumo.py
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Set JAX to run on CPU or GPU based on environment
import jax
import jax.numpy as jnp
import diffrax
import optax

# Force JAX float64 for higher precision scientific calculations if needed,
# but float32 is standard and faster. We stick to float32 here.

# =====================================================================
# 1. FitzHugh-Nagumo Equations & Solver
# =====================================================================

def get_external_current(t, I_type, I_val):
    """
    Computes external stimulus current dynamically at time t.
    Supports constant, step, sine, and pulse time series currents.
    Uses JAX where-clauses for stable tracing/JIT compilation.
    """
    constant_current = I_val
    
    # Step current: active between t=10 and t=80
    step_current = jnp.where((t >= 10.0) & (t <= 80.0), I_val, 0.0)
    
    # Sine current: oscillating around a base current
    sine_current = I_val * (1.0 + 0.5 * jnp.sin(0.2 * t))
    
    # Pulse current: periodic pulse train (period 20, active for 5)
    pulse_current = jnp.where(jnp.mod(t, 20.0) <= 5.0, I_val, 0.0)
    
    # Branch at compile-time on static string parameter
    if I_type == 'step':
        return step_current
    elif I_type == 'sine':
        return sine_current
    elif I_type == 'pulse':
        return pulse_current
    else:
        return constant_current


def fhn_vector_field(t, y, args):
    """
    FitzHugh-Nagumo Ordinary Differential Equations.
    
    dv/dt = v - v^3/3 - w + I_ext(t)
    dw/dt = (v + a - b*w) / tau
    
    Args:
        t: Time scalar
        y: State array of shape (2,) representing [v, w]
           v: Membrane potential
           w: Recovery variable
        args: Tuple of parameters (a, b, tau, I_type, I_val)
    """
    v, w = y
    a, b, tau, I_type, I_val = args
    
    I_ext = get_external_current(t, I_type, I_val)
    dv_dt = v - (v**3) / 3.0 - w + I_ext
    dw_dt = (v + a - b * w) / tau
    
    return jnp.stack([dv_dt, dw_dt])


def simulate_fhn(y0, t_span, a=0.7, b=0.8, tau=12.5, I_type='constant', I_val=0.5, rtol=1e-6, atol=1e-6):
    """
    Simulates the FHN model using the Diffrax Tsit5 adaptive solver.
    
    Args:
        y0: Initial state [v0, w0] (shape: (2,))
        t_span: Time steps at which to save results
        a, b, tau: Model parameters
        I_type: Type of external current ('constant', 'step', 'sine', 'pulse')
        I_val: Amplitude or base value of stimulus current
        rtol, atol: Relative/absolute tolerances for adaptive solver
        
    Returns:
        ys: Trajectory of shape (n_steps, 2)
    """
    # Create the ODE terms
    term = diffrax.ODETerm(fhn_vector_field)
    
    # Adaptive Tsitouras 5th order solver (highly efficient)
    solver = diffrax.Tsit5()
    
    # Configure saving time points
    saveat = diffrax.SaveAt(ts=t_span)
    
    # PID controller for adaptive stepping
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=0.05,  # initial step suggestion
        y0=jnp.asarray(y0, dtype=jnp.float32),
        args=(a, b, tau, I_type, I_val),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000
    )
    
    return sol.ys


# =====================================================================
# 2. Parameter Fitting using JAX & Optax
# =====================================================================

def fit_fhn_parameters(y0, t_span, noisy_target, I_type, I_val, lr=0.02, steps=150):
    """
    Fits the FHN parameters (a, b, tau) to match target data using gradient descent.
    
    Args:
        y0: Initial condition [v0, w0]
        t_span: Output time points
        noisy_target: Trajectory data to fit [n_steps, 2]
        I_type: Type of external current used in the target data
        I_val: Amplitude or base value of stimulus current used in target data
        lr: Learning rate for Optax Adam
        steps: Number of optimization steps
        
    Returns:
        fitted_params: Dict of fitted parameters
        loss_history: List of losses per step
        fitted_trajectory: Integrated trajectory with fitted parameters
    """
    # Ensure inputs are JAX arrays to prevent pytree structure mismatches with python lists
    y0 = jnp.asarray(y0, dtype=jnp.float32)
    noisy_target = jnp.asarray(noisy_target, dtype=jnp.float32)

    # Initial guess for parameters: a=0.5, b=0.5, tau=10.0 (true: 0.7, 0.8, 12.5)
    init_params = jnp.array([0.5, 0.5, 10.0])
    
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def loss_fn(params, y0, t_span, target, I_type, I_val):
        # Enforce parameter positivity using absolute value
        a, b, tau = jnp.abs(params)
        
        # Simulate with current parameter guesses
        term = diffrax.ODETerm(fhn_vector_field)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=t_span)
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-4)
        
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[-1],
            dt0=0.1,
            y0=y0,
            args=(a, b, tau, I_type, I_val),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=5000
        )
        
        # Mean Squared Error loss
        return jnp.mean((sol.ys - target) ** 2)

    @jax.jit
    def train_step(params, opt_state, y0, t_span, target, I_type, I_val):
        loss, grads = jax.value_and_grad(loss_fn)(params, y0, t_span, target, I_type, I_val)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params = init_params
    loss_history = []
    
    print("\nStarting Parameter Fitting Demo with Optax (JAX + Diffrax Stack)...")
    print(f"Initial guesses: a={params[0]:.2f}, b={params[1]:.2f}, tau={params[2]:.2f}")
    
    for step in range(steps):
        params, opt_state, loss = train_step(params, opt_state, y0, t_span, noisy_target, I_type, I_val)
        loss_history.append(float(loss))
        
        if step % 15 == 0 or step == steps - 1:
            a_cur, b_cur, tau_cur = np.abs(params)
            print(f"Step {step:03d} | Loss: {float(loss):.6f} | Current guesses: a={a_cur:.4f}, b={b_cur:.4f}, tau={tau_cur:.4f}")
            
    fitted_vals = np.abs(params)
    fitted_params = {"a": float(fitted_vals[0]), "b": float(fitted_vals[1]), "tau": float(fitted_vals[2])}
    
    # Simulate the fitted trajectory
    fitted_trajectory = simulate_fhn(
        y0, t_span, 
        a=fitted_params["a"], 
        b=fitted_params["b"], 
        tau=fitted_params["tau"], 
        I_type=I_type,
        I_val=I_val
    )
    
    return fitted_params, loss_history, fitted_trajectory


# =====================================================================
# 2b. Hankel Dynamic Mode Decomposition (Hankel-DMD)
# =====================================================================

def run_hankel_dmd(v_data, H, r):
    """
    Computes Hankel Dynamic Mode Decomposition (Hankel-DMD) on time-series v_data.
    
    Args:
        v_data: Time series array of shape (T,) (e.g. membrane potential v)
        H: Delay embedding dimension (number of rows of Hankel matrix)
        r: SVD truncation rank (energetic modes to keep)
        
    Returns:
        A: Truncated dynamics matrix of shape (r, r)
        eigenvalues: Complex eigenvalues of A representing Koopman modes
        s: Full spectrum of singular values from the Hankel SVD
        X: The time-shifted matrix from H[:, :-1]
        Y: The time-shifted matrix from H[:, 1:]
    """
    v_data = jnp.asarray(v_data, dtype=jnp.float32)
    T = len(v_data)
    
    if H >= T:
        raise ValueError(f"Hankel delay embedding H ({H}) must be strictly less than time series length T ({T})")
        
    K = T - H + 1
    
    # 1. Construct Hankel matrix of shape (H, K)
    # Using JAX to stack delay slices
    H_matrix = jnp.stack([v_data[i : i + K] for i in range(H)], axis=0)
    
    # 2. Slice into shifted matrices X (1 to K-1) and Y (2 to K)
    X = H_matrix[:, :-1]  # entries from first to T-1 of Hankel columns
    Y = H_matrix[:, 1:]   # entries from second to T of Hankel columns
    
    # 3. Singular Value Decomposition of X: X = U * Sigma * V^T
    U, s, V_T = jnp.linalg.svd(X, full_matrices=False)
    V = V_T.T
    
    # Bound rank r to prevent index errors
    r = min(r, U.shape[1])
    
    # 4. Truncate SVD matrices to rank r (energy modes)
    U_r = U[:, :r]             # (H, r)
    V_r = V[:, :r]             # (T-H, r)
    s_r = s[:r]                # (r,)
    
    # 5. Define truncated transition matrix A = U_r.T @ Y @ V_r @ Sigma^-1
    Sigma_inv = jnp.diag(1.0 / s_r)
    A = U_r.T @ Y @ V_r @ Sigma_inv  # (r, r)
    
    # 6. Compute complex eigenvalues of the truncated operator
    eigenvalues = jnp.linalg.eigvals(A)
    
    return A, eigenvalues, s, X, Y


# =====================================================================
# 2c. Dynamic Mode Decomposition with Control (DMDc)
# =====================================================================

def run_dmdc(v_data, u_data, H, r, p):
    """
    Computes Dynamic Mode Decomposition with Control (DMDc) on Hankel matrices.
    
    Args:
        v_data: Time series array of state (T,) (potential v)
        u_data: Time series array of time-varying control inputs (T,) (external current)
        H: Delay embedding dimension
        r: State truncation rank (autonomous states U_r)
        p: Augmented space truncation rank (state+control spaces U_p)
        
    Returns:
        A_tilde: Truncated autonomous dynamics transition operator (r, r)
        B_tilde: Truncated control input matrix (r, 1)
        eigenvalues: Complex eigenvalues of A_tilde (autonomous modes)
        s_x: Singular values of state matrix X
        s_p: Singular values of augmented state-control matrix Omega
        X, Y: Shifted Hankel matrices
        U_c: Control input matrix matching Hankel column time slices
    """
    v_data = jnp.asarray(v_data, dtype=jnp.float32)
    u_data = jnp.asarray(u_data, dtype=jnp.float32)
    T = len(v_data)
    
    if H >= T:
        raise ValueError(f"Hankel delay H ({H}) must be strictly less than time series length T ({T})")
        
    K = T - H + 1
    
    # 1. Construct Hankel matrix of state
    H_state = jnp.stack([v_data[i : i + K] for i in range(H)], axis=0)  # (H, K)
    
    # Construct control input vector matching columns (causal: input at current lead step)
    U_c = jnp.stack([u_data[i + H - 1] for i in range(K - 1)], axis=0).reshape(1, -1)  # (1, K-1)
    
    # 2. Shifted Hankel matrices X and Y
    X = H_state[:, :-1]  # (H, K-1)
    Y = H_state[:, 1:]   # (H, K-1)
    
    # 3. Construct Augmented matrix Omega = [X; U_c]
    Omega = jnp.concatenate([X, U_c], axis=0)  # (H + 1, K-1)
    
    # 4. SVD of Omega: Omega = U_p * Sigma_p * V_p^T
    U_tilde, s_p, V_p_T = jnp.linalg.svd(Omega, full_matrices=False)
    V_p = V_p_T.T
    
    p = min(p, U_tilde.shape[1])
    U_p = U_tilde[:, :p]             # (H + 1, p)
    V_p = V_p[:, :p]                 # (K-1, p)
    s_p_r = s_p[:p]                  # (p,)
    
    # Split U_p into autonomous (first H rows) and control (last 1 row)
    U_p1 = U_p[:H, :]                # (H, p)
    U_p2 = U_p[H:, :]                # (1, p)
    
    # 5. SVD of X for state projection: X = U_x * Sigma_x * V_x^T
    U_x, s_x, V_x_T = jnp.linalg.svd(X, full_matrices=False)
    
    r = min(r, U_x.shape[1])
    U_r = U_x[:, :r]                 # (H, r)
    
    # 6. Compute low-dimensional transition operators A_tilde and B_tilde
    Sigma_p_inv = jnp.diag(1.0 / s_p_r)
    
    # Formula: A_tilde = U_r.T @ Y @ V_p @ Sigma_p^-1 @ U_p1.T @ U_r
    A_tilde = U_r.T @ Y @ V_p @ Sigma_p_inv @ U_p1.T @ U_r  # (r, r)
    
    # Formula: B_tilde = U_r.T @ Y @ V_p @ Sigma_p^-1 @ U_p2.T
    B_tilde = U_r.T @ Y @ V_p @ Sigma_p_inv @ U_p2.T        # (r, 1)
    
    # 7. Koopman autonomous eigenvalues
    eigenvalues = jnp.linalg.eigvals(A_tilde)
    
    return A_tilde, B_tilde, eigenvalues, s_x, s_p, X, Y, U_c


# =====================================================================
# 3. Scientific Plotting and Visualization
# =====================================================================

def plot_results(t_span, ys, a, b, tau, I_type, I_val, y0, u_data=None, fitted_data=None, noisy_target=None, save_path=None, show_plot=True):
    """
    Generates premium-quality scientific plots of the simulation results.
    
    Args:
        t_span: Time steps
        ys: Trajectory of shape (n_steps, 2)
        a, b, tau: Model parameters
        I_type: Current simulation type
        I_val: Current amplitude
        y0: Initial conditions
        u_data: Time series array of actual external current values
        fitted_data: Optional fitted trajectory from Optax
        noisy_target: Optional noisy data used in fitting
        save_path: Path to save the plot as a PNG image
        show_plot: Whether to display the plot interactively using plt.show()
    """
    v = ys[:, 0]
    w = ys[:, 1]
    
    # Modern, sleek style settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(14, 6))
    
    # Color palette
    c_v = '#1f77b4'       # Electric blue for V
    c_w = '#ff7f0e'       # Sunset orange for W
    c_v_null = '#2ca02c'  # Green for V nullcline
    c_w_null = '#d62728'  # Red for W nullcline
    
    # --- Left Subplot: Time Series ---
    ax1 = fig.add_subplot(121)
    ax1.plot(t_span, v, label=r'Membrane Potential $v(t)$', color=c_v, linewidth=2.0)
    ax1.plot(t_span, w, label=r'Recovery Variable $w(t)$', color=c_w, linewidth=2.0)
    
    # Plot dynamic external current if provided
    if u_data is not None:
        ax1.plot(t_span, u_data, label=r'Stimulus Current $I_{ext}(t)$', color='#d62728', linewidth=1.5, linestyle=':', alpha=0.9)
    
    if noisy_target is not None:
        ax1.scatter(t_span[::5], noisy_target[::5, 0], color='black', alpha=0.3, s=8, label='Noisy Target $v_{meas}$')
    if fitted_data is not None:
        ax1.plot(t_span, fitted_data[:, 0], '--', color='#9467bd', linewidth=1.5, label='Fitted $v_{opt}$')
        
    ax1.set_title("Neural Activation Time Series", fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Time (dimensionless)", fontsize=12)
    ax1.set_ylabel("State Magnitude", fontsize=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax1.set_xlim(t_span[0], t_span[-1])
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Right Subplot: Phase Space & Nullclines ---
    ax2 = fig.add_subplot(122)
    
    # Plot nullclines using base external current I_val
    # v-nullcline: w = v - v^3/3 + I_val
    # w-nullcline: w = (v + a) / b
    v_vals = np.linspace(np.min(v) - 0.5, np.max(v) + 0.5, 400)
    v_nullcline = v_vals - (v_vals**3) / 3.0 + I_val
    w_nullcline = (v_vals + a) / b
    
    ax2.plot(v_vals, v_nullcline, '--', color=c_v_null, alpha=0.8, linewidth=1.5, label=r'$v$-nullcline ($dv/dt=0$)')
    ax2.plot(v_vals, w_nullcline, '--', color=c_w_null, alpha=0.8, linewidth=1.5, label=r'$w$-nullcline ($dw/dt=0$)')
    
    # Plot trajectories
    ax2.plot(v, w, color='#9467bd', linewidth=2.5, label='System Trajectory')
    ax2.scatter(y0[0], y0[1], color='red', s=50, zorder=5, label=r'Initial Condition $(v_0, w_0)$')
    
    # Highlight fixed point (intersection of nullclines at base external current I_val)
    # The fixed point satisfies: (v + a)/b = v - v^3/3 + I_val
    # We can plot the intersection cleanly by finding the numerical root
    from scipy.optimize import fsolve
    fp_func = lambda x: x - (x**3)/3.0 - (x + a)/b + I_val
    fp_v = float(fsolve(fp_func, 0.0)[0])
    fp_w = (fp_v + a) / b
    ax2.scatter(fp_v, fp_w, color='black', marker='*', s=120, zorder=6, label=f'Fixed Point ({fp_v:.2f}, {fp_w:.2f})')
    
    ax2.set_title("Phase Portrait & Nullclines", fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel(r"Membrane Potential $v$", fontsize=12)
    ax2.set_ylabel(r"Recovery Variable $w$", fontsize=12)
    ax2.set_ylim(np.min(w) - 0.3, np.max(w) + 0.3)
    ax2.set_xlim(np.min(v) - 0.3, np.max(v) + 0.3)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle(f"FitzHugh-Nagumo Model Dynamics\n(a={a:.2f}, b={b:.2f}, \u03c4={tau:.2f}, Current={I_type} ({I_val:.2f}))", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_dmdc_results(s_x, s_p, eigenvalues, B_tilde, r, H, p, save_path=None, show_plot=True):
    """
    Visualizes DMDc results:
    1. Singular value decay of X and Omega (augmented space).
    2. Koopman eigenvalues of A_tilde on unit circle (base autonomous dynamics).
    3. Input control mode sensitivities of B_tilde (influence magnitude).
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 5.5))
    
    # --- 1. SVD Energy Spectra ---
    ax1 = fig.add_subplot(131)
    ax1.semilogy(s_x, 'o-', color='#1f77b4', markersize=4, label='State X Spectrum')
    ax1.semilogy(s_p, 's--', color='#9467bd', markersize=4, label=r'Augmented $\Omega$ Spectrum')
    ax1.axvline(x=r-1, color='#d62728', linestyle=':', label=f'State Truncation r={r}')
    ax1.axvline(x=p-1, color='#2ca02c', linestyle='-.', label=f'Augmented Truncation p={p}')
    ax1.set_title("SVD Energy Spectra Decay", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Singular Value Index", fontsize=11)
    ax1.set_ylabel("Singular Value Magnitude", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, which="both", alpha=0.5)
    
    # --- 2. Koopman eigenvalues (Base autonomous Dynamics A) ---
    ax2 = fig.add_subplot(132)
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.7, label='Unit Circle')
    ax2.scatter(eigenvalues.real, eigenvalues.imag, color='#2ca02c', edgecolor='black', s=70, zorder=5, label='Autonomous modes')
    ax2.set_title(f"Intrinsic Koopman Spectrum (r={r})", fontsize=12, fontweight='bold')
    ax2.set_xlabel(r"Real Part $\Re(\lambda)$", fontsize=11)
    ax2.set_ylabel(r"Imaginary Part $\Im(\lambda)$", fontsize=11)
    ax2.grid(True, alpha=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.legend(frameon=True, loc='upper right')
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_ylim(-1.4, 1.4)
    
    # --- 3. Input Mode Control Sensitivities (B) ---
    ax3 = fig.add_subplot(133)
    b_magnitudes = np.abs(np.squeeze(B_tilde))
    indices = np.arange(len(b_magnitudes))
    ax3.bar(indices, b_magnitudes, color='#ff7f0e', edgecolor='black', alpha=0.85, width=0.6)
    ax3.set_title("Control Input Sensitivity (|B|)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Subspace Mode Index", fontsize=11)
    ax3.set_ylabel("Influence Magnitude", fontsize=11)
    ax3.set_xticks(indices)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(f"Dynamic Mode Decomposition with Control (DMDc) Analysis\n(Base Dynamics A_tilde: {r}x{r} | Input Coupling B_tilde: {r}x1 | Delays H={H})", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        dmdc_save_path = save_path.replace(".png", "_dmdc.png")
        plt.savefig(dmdc_save_path, dpi=300)
        print(f"Saved DMDc visualization plot to {dmdc_save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_dmd_results(s, eigenvalues, r, H, save_path=None, show_plot=True):
    """
    Plots the SVD Singular Value Spectrum and Koopman eigenvalues on the complex unit circle.
    """
    # Modern, sleek style settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left Plot: Singular Value Spectrum ---
    ax1.semilogy(s, 'o-', color='#1f77b4', linewidth=2.0, markersize=5, label='Singular Values')
    ax1.axvline(x=r-1, color='#d62728', linestyle='--', linewidth=1.5, label=f'Truncation Rank r={r}')
    ax1.set_title("SVD Singular Value Spectrum (Energy Decay)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Singular Value Index", fontsize=11)
    ax1.set_ylabel("Singular Value Magnitude (log scale)", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # --- Right Plot: Complex Plane eigenvalues and unit circle ---
    # Plot Unit Circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.7, label='Unit Circle')
    
    # Plot DMD eigenvalues
    ax2.scatter(eigenvalues.real, eigenvalues.imag, color='#2ca02c', edgecolor='black', s=80, zorder=5, label='DMD Eigenvalues')
    
    ax2.set_title(f"DMD Koopman Spectrum (Complex Plane, r={r})", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel(r"Real Part $\Re(\lambda)$", fontsize=11)
    ax2.set_ylabel(r"Imaginary Part $\Im(\lambda)$", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
    ax2.set_aspect('equal')
    
    # Legend placement
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    # Limit range to showcase the unit circle stability
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    
    plt.suptitle(f"Hankel Dynamic Mode Decomposition (Hankel-DMD) Analysis\n(Delay coordinates H={H}, Truncated state space r={r})", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        dmd_save_path = save_path.replace(".png", "_dmd.png")
        plt.savefig(dmd_save_path, dpi=300)
        print(f"Saved DMD visualization plot to {dmd_save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()


# =====================================================================
# 4. Main Script Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="FitzHugh-Nagumo Neural Model Simulator and Parameter Estimator")
    
    # Simulation settings
    parser.add_argument('--v0', type=float, default=-1.5, help="Initial membrane potential (default: -1.5)")
    parser.add_argument('--w0', type=float, default=-0.5, help="Initial recovery variable (default: -0.5)")
    parser.add_argument('--a', type=float, default=0.7, help="Parameter a (default: 0.7)")
    parser.add_argument('--b', type=float, default=0.8, help="Parameter b (default: 0.8)")
    parser.add_argument('--tau', type=float, default=12.5, help="Time constant tau (default: 12.5)")
    parser.add_argument('--I', type=float, default=0.5, help="Constant external current amplitude/value I (default: 0.5)")
    parser.add_argument('--I-type', type=str, default='constant', choices=['constant', 'step', 'sine', 'pulse'],
                        help="Type of dynamic external current (constant, step, sine, pulse) (default: constant)")
    
    # Hankel-DMD/DMDc settings
    parser.add_argument('--dmd', action='store_true', help="Run Hankel Dynamic Mode Decomposition (Hankel-DMD)")
    parser.add_argument('--dmdc', action='store_true', help="Run Dynamic Mode Decomposition with Control (DMDc)")
    parser.add_argument('--dmd-H', type=int, default=50, help="Delay embedding dimension H for Hankel matrix (default: 50)")
    parser.add_argument('--dmd-r', type=int, default=10, help="Truncation rank r for state projection subspace (default: 10)")
    parser.add_argument('--dmd-p', type=int, default=15, help="Truncation rank p for DMDc augmented state-control matrix (default: 15)")
    
    # Time settings
    parser.add_argument('--t-max', type=float, default=100.0, help="Total simulation time (default: 100.0)")
    parser.add_argument('--dt', type=float, default=0.1, help="Sampling time step (default: 0.1)")
    
    # Output and demo options
    parser.add_argument('--output', type=str, default=None, help="Output CSV filename to save time series (e.g. fhn_data.csv)")
    parser.add_argument('--no-plot', action='store_true', help="Disable matplotlib plotting")
    parser.add_argument('--save-plot', type=str, default=None, help="Save the plot as a PNG image file (e.g. fhn_plot.png)")
    parser.add_argument('--fit-demo', action='store_true', help="Run parameters fitting demonstration using Optax")
    
    args = parser.parse_args()
    
    # Setup initial state and time span
    y0 = [args.v0, args.w0]
    n_steps = int(args.t_max / args.dt) + 1
    t_span = jnp.linspace(0.0, args.t_max, n_steps)
    
    # Pre-generate external current time series for visualization and DMDc
    u_data = jnp.array([get_external_current(t, args.I_type, args.I) for t in t_span])
    
    # 1. Run Simulation
    print(f"Simulating FitzHugh-Nagumo model...")
    print(f"Parameters: a={args.a}, b={args.b}, tau={args.tau}, Current={args.I_type} (amplitude={args.I})")
    print(f"Time span: [0, {args.t_max}] with dt={args.dt} ({n_steps} points)")
    
    ys = simulate_fhn(
        y0, t_span, 
        a=args.a, b=args.b, tau=args.tau, I_type=args.I_type, I_val=args.I
    )
    
    # 2. Optional Optax Parameter Fitting Demonstration
    fitted_trajectory = None
    noisy_target = None
    true_a, true_b, true_tau = args.a, args.b, args.tau
    
    if args.fit_demo:
        # Create noisy target data from the simulation
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, ys.shape) * 0.08
        noisy_target = ys + noise
        
        # Fit parameters
        fitted_params, loss_history, fitted_trajectory = fit_fhn_parameters(
            y0, t_span, noisy_target, args.I_type, args.I, lr=0.03, steps=150
        )
        
        print("\nOptimization Complete!")
        print(f"Target Values: a={true_a:.4f}, b={true_b:.4f}, tau={true_tau:.4f}")
        print(f"Fitted Values: a={fitted_params['a']:.4f}, b={fitted_params['b']:.4f}, tau={fitted_params['tau']:.4f}")
        print(f"Absolute error: a_err={abs(fitted_params['a']-true_a):.4f}, b_err={abs(fitted_params['b']-true_b):.4f}, tau_err={abs(fitted_params['tau']-true_tau):.4f}")
        
    # 3. Save Data to CSV
    if args.output:
        import csv
        filepath = args.output
        print(f"\nSaving time series data to {filepath}...")
        try:
            with open(filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = ['time', 'v_potential', 'w_recovery', 'I_ext']
                if args.fit_demo:
                    header += ['v_measured', 'w_measured', 'v_fitted', 'w_fitted']
                writer.writerow(header)
                
                for i in range(len(t_span)):
                    row = [float(t_span[i]), float(ys[i, 0]), float(ys[i, 1]), float(u_data[i])]
                    if args.fit_demo:
                        row += [float(noisy_target[i, 0]), float(noisy_target[i, 1]),
                                float(fitted_trajectory[i, 0]), float(fitted_trajectory[i, 1])]
                    writer.writerow(row)
            print(f"Successfully wrote {len(t_span)} steps of time series data to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            
    # 4. Hankel Dynamic Mode Decomposition (Hankel-DMD)
    if args.dmd:
        print(f"\nRunning Hankel-DMD Analysis on potential v...")
        print(f"Hankel Matrix parameters: H={args.dmd_H}, Truncation Rank r={args.dmd_r}")
        
        try:
            A_matrix, dmd_eigenvalues, s_vals, dmd_X, dmd_Y = run_hankel_dmd(
                ys[:, 0], H=args.dmd_H, r=args.dmd_r
            )
            
            print("Hankel-DMD Complete!")
            print(f"Shifted Hankel X shape: {dmd_X.shape}")
            print(f"Shifted Hankel Y shape: {dmd_Y.shape}")
            print(f"Truncated Dynamic Matrix A shape: {A_matrix.shape}")
            print(f"Top 5 Singular Values: {s_vals[:5]}")
            print(f"Koopman Eigenvalues (first 5):\n{dmd_eigenvalues[:5]}")
            
            if args.output:
                dmd_output_path = args.output.replace(".csv", "_dmd_A.csv")
                np.savetxt(dmd_output_path, A_matrix, delimiter=",")
                print(f"Saved truncated transition matrix A to {dmd_output_path}")
                
            if not args.no_plot or args.save_plot:
                print("Generating DMD matplotlib spectrum plots...")
                plot_dmd_results(
                    s_vals, dmd_eigenvalues, r=args.dmd_r, H=args.dmd_H,
                    save_path=args.save_plot, show_plot=not args.no_plot
                )
        except Exception as e:
            print(f"Error running DMD: {e}")

    # 4b. Dynamic Mode Decomposition with Control (DMDc)
    if args.dmdc:
        print(f"\nRunning Dynamic Mode Decomposition with Control (DMDc)...")
        print(f"Hankel parameters: H={args.dmd_H} | Truncation Ranks: state r={args.dmd_r}, augmented p={args.dmd_p}")
        
        try:
            A_tilde, B_tilde, dmdc_eigenvalues, s_x, s_p, dmdc_X, dmdc_Y, dmdc_U = run_dmdc(
                ys[:, 0], u_data, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
            )
            
            print("DMDc Complete!")
            print(f"Augmented state-input matrix Omega shape: ({dmdc_X.shape[0] + dmdc_U.shape[0]}, {dmdc_X.shape[1]})")
            print(f"Autonomous transition A_tilde shape: {A_tilde.shape}")
            print(f"Control coupling B_tilde shape: {B_tilde.shape}")
            print(f"Top 5 Intrinsic Singular Values: {s_x[:5]}")
            print(f"Intrinsic Koopman Eigenvalues (first 5):\n{dmdc_eigenvalues[:5]}")
            
            if args.output:
                dmdc_A_path = args.output.replace(".csv", "_dmdc_A.csv")
                dmdc_B_path = args.output.replace(".csv", "_dmdc_B.csv")
                np.savetxt(dmdc_A_path, A_tilde, delimiter=",")
                np.savetxt(dmdc_B_path, B_tilde, delimiter=",")
                print(f"Saved autonomous operator A_tilde to {dmdc_A_path}")
                print(f"Saved control operator B_tilde to {dmdc_B_path}")
                
            if not args.no_plot or args.save_plot:
                print("Generating DMDc matplotlib spectra and control plots...")
                plot_dmdc_results(
                    s_x, s_p, dmdc_eigenvalues, B_tilde, r=args.dmd_r, H=args.dmd_H, p=args.dmd_p,
                    save_path=args.save_plot, show_plot=not args.no_plot
                )
        except Exception as e:
            print(f"Error running DMDc: {e}")

    # 5. Visualization
    if not args.no_plot or args.save_plot:
        print("\nGenerating matplotlib visualization...")
        plot_results(
            t_span, ys, 
            a=args.a, b=args.b, tau=args.tau, I_type=args.I_type, I_val=args.I, 
            y0=y0, 
            u_data=u_data,
            fitted_data=fitted_trajectory, 
            noisy_target=noisy_target,
            save_path=args.save_plot,
            show_plot=not args.no_plot
        )


if __name__ == '__main__':
    main()
