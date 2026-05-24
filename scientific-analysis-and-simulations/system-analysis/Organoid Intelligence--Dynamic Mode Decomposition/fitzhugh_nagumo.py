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

def fhn_vector_field(t, y, args):
    """
    FitzHugh-Nagumo Ordinary Differential Equations.
    
    dv/dt = v - v^3/3 - w + I_ext
    dw/dt = (v + a - b*w) / tau
    
    Args:
        t: Time scalar
        y: State array of shape (2,) representing [v, w]
           v: Membrane potential
           w: Recovery variable
        args: Tuple of parameters (a, b, tau, I_ext)
    """
    v, w = y
    a, b, tau, I_ext = args
    
    dv_dt = v - (v**3) / 3.0 - w + I_ext
    dw_dt = (v + a - b * w) / tau
    
    return jnp.stack([dv_dt, dw_dt])


def simulate_fhn(y0, t_span, a=0.7, b=0.8, tau=12.5, I_ext=0.5, rtol=1e-6, atol=1e-6):
    """
    Simulates the FHN model using the Diffrax Tsit5 adaptive solver.
    
    Args:
        y0: Initial state [v0, w0] (shape: (2,))
        t_span: Time steps at which to save results
        a, b, tau: Model parameters
        I_ext: Constant external stimulus current
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
        args=(a, b, tau, I_ext),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000
    )
    
    return sol.ys


# =====================================================================
# 2. Parameter Fitting using JAX & Optax
# =====================================================================

def fit_fhn_parameters(y0, t_span, noisy_target, I_ext, lr=0.02, steps=150):
    """
    Fits the FHN parameters (a, b, tau) to match target data using gradient descent.
    
    Args:
        y0: Initial condition [v0, w0]
        t_span: Output time points
        noisy_target: Trajectory data to fit [n_steps, 2]
        I_ext: Stimulus current used in the target data
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
    # We optimize log(params) to enforce positivity during training!
    init_params = jnp.array([0.5, 0.5, 10.0])
    
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def loss_fn(params, y0, t_span, target, I_ext):
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
            args=(a, b, tau, I_ext),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=5000
        )
        
        # Mean Squared Error loss
        return jnp.mean((sol.ys - target) ** 2)

    @jax.jit
    def train_step(params, opt_state, y0, t_span, target, I_ext):
        loss, grads = jax.value_and_grad(loss_fn)(params, y0, t_span, target, I_ext)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params = init_params
    loss_history = []
    
    print("\nStarting Parameter Fitting Demo with Optax (JAX + Diffrax Stack)...")
    print(f"Initial guesses: a={params[0]:.2f}, b={params[1]:.2f}, tau={params[2]:.2f}")
    
    for step in range(steps):
        params, opt_state, loss = train_step(params, opt_state, y0, t_span, noisy_target, I_ext)
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
        I_ext=I_ext
    )
    
    return fitted_params, loss_history, fitted_trajectory


# =====================================================================
# 3. Scientific Plotting and Visualization
# =====================================================================

def plot_results(t_span, ys, a, b, tau, I_ext, y0, fitted_data=None, noisy_target=None, save_path=None, show_plot=True):
    """
    Generates premium-quality scientific plots of the simulation results.
    
    Args:
        t_span: Time steps
        ys: Trajectory of shape (n_steps, 2)
        a, b, tau, I_ext: Parameters
        y0: Initial conditions
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
    
    # Plot nullclines
    # v-nullcline: w = v - v^3/3 + I_ext
    # w-nullcline: w = (v + a) / b
    v_vals = np.linspace(np.min(v) - 0.5, np.max(v) + 0.5, 400)
    v_nullcline = v_vals - (v_vals**3) / 3.0 + I_ext
    w_nullcline = (v_vals + a) / b
    
    ax2.plot(v_vals, v_nullcline, '--', color=c_v_null, alpha=0.8, linewidth=1.5, label=r'$v$-nullcline ($dv/dt=0$)')
    ax2.plot(v_vals, w_nullcline, '--', color=c_w_null, alpha=0.8, linewidth=1.5, label=r'$w$-nullcline ($dw/dt=0$)')
    
    # Plot trajectories
    ax2.plot(v, w, color='#9467bd', linewidth=2.5, label='System Trajectory')
    ax2.scatter(y0[0], y0[1], color='red', s=50, zorder=5, label=r'Initial Condition $(v_0, w_0)$')
    
    # Highlight fixed point (intersection of nullclines)
    # The fixed point satisfies: (v + a)/b = v - v^3/3 + I_ext
    # We can plot the intersection cleanly by finding the numerical root
    from scipy.optimize import fsolve
    fp_func = lambda x: x - (x**3)/3.0 - (x + a)/b + I_ext
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
    
    plt.suptitle(f"FitzHugh-Nagumo Model Dynamics\n(a={a:.2f}, b={b:.2f}, \u03c4={tau:.2f}, I={I_ext:.2f})", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization plot to {save_path}")
        
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
    parser.add_argument('--I', type=float, default=0.5, help="Constant external current I (default: 0.5)")
    
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
    
    # 1. Run Simulation
    print(f"Simulating FitzHugh-Nagumo model...")
    print(f"Parameters: a={args.a}, b={args.b}, tau={args.tau}, I_ext={args.I}")
    print(f"Time span: [0, {args.t_max}] with dt={args.dt} ({n_steps} points)")
    
    ys = simulate_fhn(
        y0, t_span, 
        a=args.a, b=args.b, tau=args.tau, I_ext=args.I
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
            y0, t_span, noisy_target, args.I, lr=0.03, steps=150
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
                header = ['time', 'v_potential', 'w_recovery']
                if args.fit_demo:
                    header += ['v_measured', 'w_measured', 'v_fitted', 'w_fitted']
                writer.writerow(header)
                
                for i in range(len(t_span)):
                    row = [float(t_span[i]), float(ys[i, 0]), float(ys[i, 1])]
                    if args.fit_demo:
                        row += [float(noisy_target[i, 0]), float(noisy_target[i, 1]),
                                float(fitted_trajectory[i, 0]), float(fitted_trajectory[i, 1])]
                    writer.writerow(row)
            print(f"Successfully wrote {len(t_span)} steps of time series data to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            
    # 4. Visualization
    if not args.no_plot or args.save_plot:
        print("\nGenerating matplotlib visualization...")
        plot_results(
            t_span, ys, 
            a=args.a, b=args.b, tau=args.tau, I_ext=args.I, 
            y0=y0, 
            fitted_data=fitted_trajectory, 
            noisy_target=noisy_target,
            save_path=args.save_plot,
            show_plot=not args.no_plot
        )


if __name__ == '__main__':
    main()
