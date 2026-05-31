"""
Bilinear DMDc operator recursive validation and trajectory reconstruction testing.
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from dynamics import get_external_current, run_dmdc
from simulation import simulate_fhn, simulate_fhn_batch
from plotting import plot_reconstruction

def main():
    """Simulates FHN dynamics, learns bilinear operators, and performs recursive trajectory reconstruction."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Bilinear DMDc Recursive Trajectory Reconstruction Validator")
    
    parser.add_argument('--v0', type=float, default=-1.5, help="Initial membrane potential")
    parser.add_argument('--w0', type=float, default=-0.5, help="Initial recovery variable")
    parser.add_argument('--a', type=float, default=0.7, help="Parameter a")
    parser.add_argument('--b', type=float, default=0.8, help="Parameter b")
    parser.add_argument('--tau', type=float, default=12.5, help="Time constant tau")
    parser.add_argument('--I', type=float, default=0.5, help="Constant external current amplitude")
    parser.add_argument('--I-type', type=str, default='sine', choices=['constant', 'step', 'sine', 'pulse'],
                        help="Type of dynamic external current")
    
    parser.add_argument('--batch', action='store_true', help="Run in multi-trajectory batch mode")
    parser.add_argument('--batch-size', type=int, default=5, help="Number of trajectories in batch")
    
    parser.add_argument('--dmd-H', type=int, default=70, help="Delay embedding dimension H")
    parser.add_argument('--dmd-r', type=int, default=12, help="Truncation rank r for state projection")
    parser.add_argument('--dmd-p', type=int, default=20, help="Truncation rank p for augmented space")
    
    parser.add_argument('--t-max', type=float, default=100.0, help="Total simulation time")
    parser.add_argument('--dt', type=float, default=0.1, help="Sampling time step")
    
    parser.add_argument('--no-plot', action='store_true', help="Disable matplotlib plotting")
    parser.add_argument('--save-plot', type=str, default=os.path.join(plots_dir, 'fhn_reconstruction.png'), 
                        help="Save the reconstruction plot as a PNG file")
    
    args = parser.parse_args()
    
    n_steps = int(args.t_max / args.dt) + 1
    t_span = jnp.linspace(0.0, args.t_max, n_steps)
    
    ys_single = None
    u_data = None
    
    if args.batch:
        key = jax.random.PRNGKey(101)
        key1, key2, key3 = jax.random.split(key, 3)
        v0s = jax.random.uniform(key1, (args.batch_size,), minval=-2.0, maxval=1.0)
        w0s = jax.random.uniform(key2, (args.batch_size,), minval=-1.0, maxval=0.5)
        y0_batch = jnp.stack([v0s, w0s], axis=1)
        I_val_batch = jax.random.uniform(key3, (args.batch_size,), minval=0.2, maxval=1.2)
        
        print(f"\n[Batch Mode] Simulating {args.batch_size} FHN trajectories in parallel using JAX vmap...")
        ys = simulate_fhn_batch(
            y0_batch, t_span, args.I_type, I_val_batch,
            a=args.a, b=args.b, tau=args.tau
        )
        
        u_data_batch = jnp.stack([
            jnp.array([get_external_current(t, args.I_type, iv) for t in t_span])
            for iv in I_val_batch
        ], axis=0)
        
        u_data = u_data_batch[0]
        ys_single = ys[0]
    else:
        y0 = [args.v0, args.w0]
        u_data = jnp.array([get_external_current(t, args.I_type, args.I) for t in t_span])
        
        print(f"Simulating FitzHugh-Nagumo model...")
        ys_single = simulate_fhn(
            y0, t_span, 
            a=args.a, b=args.b, tau=args.tau, I_type=args.I_type, I_val=args.I
        )
    
    print("\nRunning Bilinear DMDc operator learning...")
    if args.batch:
        A_tilde, B_tilde, C_tilde, _, _, _, _, _, _, Ur = run_dmdc(
            ys[:, :, 0], u_data_batch, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
        )
    else:
        A_tilde, B_tilde, C_tilde, _, _, _, _, _, _, Ur = run_dmdc(
            ys_single[:, 0], u_data, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
        )
        
    print("Executing recursive forward pass prediction in subspace...")
    v_true = ys_single[:, 0]
    u_true = u_data
    T = len(v_true)
    H = args.dmd_H
    K = T - H + 1
    
    x0 = v_true[:H]
    z0 = Ur.T @ x0
    
    z_list = [z0]
    z_t = z0
    
    for t in range(K - 1):
        u_t = u_true[t + H - 1]
        z_next = A_tilde @ z_t + B_tilde[:, 0] * u_t + C_tilde @ z_t * u_t
        z_list.append(z_next)
        z_t = z_next
        
    Z = jnp.stack(z_list, axis=1)
    X_pred = Ur @ Z
    
    v_pred = jnp.concatenate([X_pred[0, :], X_pred[1:, -1]])
    
    mse = float(np.mean((v_true - v_pred) ** 2))
    print(f"\nBilinear DMDc Trajectory Reconstruction Complete!")
    print(f"Reconstruction Mean Squared Error (MSE): {mse:.6e}")
    
    if not args.no_plot or args.save_plot:
        print("Generating True vs Predicted scientific overlay plot...")
        plot_reconstruction(
            t_span, v_true, v_pred,
            save_path=args.save_plot, show_plot=not args.no_plot
        )

if __name__ == '__main__':
    main()
