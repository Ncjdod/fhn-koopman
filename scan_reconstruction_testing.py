"""
Bilinear DMDc recursive trajectory validation using jax.lax.scan.
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dynamics import run_dmdc, get_external_current
from simulation import simulate_fhn, simulate_fhn_batch

def predict_trajectory_scan(v_true, u_true, A_tilde, B_tilde, C_tilde, Ur, H):
    """Recursively reconstructs the potential trajectory using jax.lax.scan."""
    T = len(v_true)
    K = T - H + 1
    x0 = v_true[:H]
    z0 = Ur.T @ x0
    
    u_seq = u_true[H-1:-1]
    
    def step_fn(z_t, u_t):
        z_next = A_tilde @ z_t + B_tilde[:, 0] * u_t + C_tilde @ z_t * u_t
        return z_next, z_next
        
    _, z_predicted = jax.lax.scan(step_fn, z0, u_seq)
    Z = jnp.concatenate([z0[jnp.newaxis, :], z_predicted], axis=0).T
    X_pred = Ur @ Z
    v_pred = jnp.concatenate([X_pred[0, :], X_pred[1:, -1]])
    return v_pred

def main():
    """Simulates FHN dynamic data, learns base operators, and performs recursive reconstruction using lax.scan."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Bilinear DMDc lax.scan trajectory validator")
    parser.add_argument('--batch', action='store_true', help="Run in multi-trajectory batch mode")
    parser.add_argument('--batch-size', type=int, default=8, help="Number of trajectories in batch")
    parser.add_argument('--dmd-H', type=int, default=70, help="Delay embedding dimension H")
    parser.add_argument('--dmd-r', type=int, default=12, help="Truncation rank r")
    parser.add_argument('--dmd-p', type=int, default=20, help="Truncation rank p")
    parser.add_argument('--t-max', type=float, default=100.0, help="Simulation time")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step")
    parser.add_argument('--I-type', type=str, default='sine', help="Dynamic current profile type")
    parser.add_argument('--no-plot', action='store_true', help="Disable matplotlib plotting")
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
        
        print(f"Simulating {args.batch_size} FHN trajectories in parallel...")
        ys = simulate_fhn_batch(y0_batch, t_span, args.I_type, I_val_batch)
        
        u_data_batch = jnp.stack([
            jnp.array([get_external_current(t, args.I_type, iv) for t in t_span])
            for iv in I_val_batch
        ], axis=0)
        
        u_data = u_data_batch[0]
        ys_single = ys[0]
    else:
        y0 = [-1.5, -0.5]
        I_val = 0.5
        u_data = jnp.array([get_external_current(t, args.I_type, I_val) for t in t_span])
        
        print("Simulating single FitzHugh-Nagumo model trajectory...")
        ys_single = simulate_fhn(y0, t_span, I_type=args.I_type, I_val=I_val)
        
    print("Learning base transition operators with standard Bilinear DMDc...")
    if args.batch:
        A_tilde, B_tilde, C_tilde, _, _, _, _, _, _, Ur = run_dmdc(
            ys[:, :, 0], u_data_batch, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
        )
    else:
        A_tilde, B_tilde, C_tilde, _, _, _, _, _, _, Ur = run_dmdc(
            ys_single[:, 0], u_data, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
        )
        
    print("Executing recursive forward pass prediction using jax.lax.scan...")
    v_true = ys_single[:, 0]
    u_true = u_data
    
    v_pred = predict_trajectory_scan(v_true, u_true, A_tilde, B_tilde, C_tilde, Ur, args.dmd_H)
    
    mse = float(np.mean((v_true - v_pred) ** 2))
    print(f"\nBilinear DMDc Trajectory lax.scan Reconstruction Complete!")
    print(f"Reconstruction Mean Squared Error (MSE): {mse:.6e}")
    
    if not args.no_plot:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig = plt.figure(figsize=(12, 6))
        
        plt.plot(t_span, v_true, label='Ground Truth FHN Simulation', color='#1f77b4', linewidth=2.0)
        plt.plot(t_span, v_pred, '--', label='Bilinear DMDc lax.scan Prediction', color='#d62728', linewidth=2.0)
        
        plt.title(f"Bilinear DMDc base operator lax.scan Reconstruction\n(Reconstruction MSE: {mse:.6e})", 
                  fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("Time (dimensionless)", fontsize=12)
        plt.ylabel("Membrane Potential v", fontsize=12)
        plt.xlim(t_span[0], t_span[-1])
        plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, 'fhn_scan_reconstruction.png')
        plt.savefig(save_path, dpi=300)
        print(f"Saved trajectory reconstruction plot to {save_path}")
        plt.close()

if __name__ == '__main__':
    main()
