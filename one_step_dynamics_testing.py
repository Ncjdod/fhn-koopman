"""
Koopman operator one-step and short-horizon MPC suitability testing script in JAX.
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dynamics import run_dmdc, get_external_current
from simulation import simulate_fhn_batch

def make_hankel_batch(v_batch, H, K):
    """Computes Hankel matrices for a batch of trajectories using vmap."""
    vmap_slice = jax.vmap(lambda i: jax.lax.dynamic_slice_in_dim(v_batch, i, K, axis=1), in_axes=0, out_axes=1)
    return vmap_slice(jnp.arange(H))

def predict_n_steps_batch(Z_batch, Uc_batch, A, B, C, n):
    """Predicts n-steps ahead recursively for a batch of trajectories."""
    L = Z_batch.shape[2]
    z_pred = Z_batch[:, :, :-n]
    for i in range(n):
        u_curr = Uc_batch[:, i : L - n + i]
        Az = jnp.einsum('ij,mjk->mik', A, z_pred)
        Bu = jnp.einsum('i,mk->mik', B[:, 0], u_curr)
        Cz = jnp.einsum('ij,mjk->mik', C, z_pred)
        Czu = jnp.einsum('mik,mk->mik', Cz, u_curr)
        z_pred = Az + Bu + Czu
    z_target = Z_batch[:, :, n:]
    return z_pred, z_target

def main():
    """Simulates FHN data, learns operators, and evaluates short-horizon prediction errors in JAX."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Koopman One-Step & MPC Validator")
    parser.add_argument('--train-batch', type=int, default=8)
    parser.add_argument('--test-batch', type=int, default=50)
    parser.add_argument('--dmd-H', type=int, default=70)
    parser.add_argument('--dmd-r', type=int, default=12)
    parser.add_argument('--dmd-p', type=int, default=20)
    parser.add_argument('--t-max', type=float, default=100.0)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    
    n_steps = int(args.t_max / args.dt) + 1
    t_span = jnp.linspace(0.0, args.t_max, n_steps)
    
    print(f"Simulating {args.train_batch} training trajectories...")
    key = jax.random.PRNGKey(202)
    key1, key2, key3 = jax.random.split(key, 3)
    v0s_tr = jax.random.uniform(key1, (args.train_batch,), minval=-2.0, maxval=1.0)
    w0s_tr = jax.random.uniform(key2, (args.train_batch,), minval=-1.0, maxval=0.5)
    y0_tr = jnp.stack([v0s_tr, w0s_tr], axis=1)
    I_val_tr = jax.random.uniform(key3, (args.train_batch,), minval=0.2, maxval=1.2)
    
    ys_tr = simulate_fhn_batch(y0_tr, t_span, 'sine', I_val_tr)
    u_tr = jax.vmap(lambda iv: get_external_current(t_span, 'sine', iv))(I_val_tr)
    
    print("Learning Bilinear DMDc operators...")
    A_tilde, B_tilde, C_tilde, _, _, _, _, _, _, Ur = run_dmdc(
        ys_tr[:, :, 0], u_tr, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
    )
    
    print(f"Simulating {args.test_batch} completely unseen test trajectories...")
    key_t = jax.random.PRNGKey(909)
    key_t1, key_t2, key_t3 = jax.random.split(key_t, 3)
    v0s_te = jax.random.uniform(key_t1, (args.test_batch,), minval=-2.0, maxval=1.0)
    w0s_te = jax.random.uniform(key_t2, (args.test_batch,), minval=-1.0, maxval=0.5)
    y0_te = jnp.stack([v0s_te, w0s_te], axis=1)
    I_val_te = jax.random.uniform(key_t3, (args.test_batch,), minval=0.2, maxval=1.2)
    
    ys_te = simulate_fhn_batch(y0_te, t_span, 'sine', I_val_te)
    u_te = jax.vmap(lambda iv: get_external_current(t_span, 'sine', iv))(I_val_te)
    
    print("Evaluating short-horizon prediction errors on test batch...")
    K = n_steps - args.dmd_H + 1
    
    X_batch = make_hankel_batch(ys_te[:, :, 0], args.dmd_H, K)
    X_m_batch = X_batch[:, :, :-1]
    Z_batch = jnp.einsum('ij,mjk->mik', Ur.T, X_m_batch)
    Uc_batch = u_te[:, args.dmd_H - 1 : args.dmd_H + K - 2]
    
    subspace_errors = []
    potential_proj_errors = []
    potential_act_errors = []
    
    for n in range(1, 6):
        z_preds_all, z_targets_all = predict_n_steps_batch(Z_batch, Uc_batch, A_tilde, B_tilde, C_tilde, n)
        
        v_preds_all = jnp.einsum('j,mjk->mk', Ur[0, :], z_preds_all)
        v_targets_reconstructed = jnp.einsum('j,mjk->mk', Ur[0, :], z_targets_all)
        v_targets_actual = X_m_batch[:, 0, n:]
        
        sub_err = float(jnp.mean((z_preds_all - z_targets_all) ** 2))
        pot_proj_err = float(jnp.mean((v_preds_all - v_targets_reconstructed) ** 2))
        pot_act_err = float(jnp.mean((v_preds_all - v_targets_actual) ** 2))
        
        subspace_errors.append(sub_err)
        potential_proj_errors.append(pot_proj_err)
        potential_act_errors.append(pot_act_err)
        
    print("\nShort-Horizon Prediction Errors on Unseen Test Dataset:")
    for n in range(5):
        print(f"Horizon Step {n+1:d} | Subspace MSE: {subspace_errors[n]:.6e} | Projected Potential MSE: {potential_proj_errors[n]:.6e} | Actual Potential MSE: {potential_act_errors[n]:.6e}")
        
    print("\nMPC Suitability Analysis:")
    if potential_act_errors[0] < 1e-6:
        print("  - One-step potential prediction error is exceptionally small (< 10^-6).")
        print("  - The Bilinear DMDc model captures the one-step dynamics perfectly.")
    else:
        print(f"  - One-step potential prediction error is: {potential_act_errors[0]:.6e}")
        
    if potential_act_errors[4] < 1e-4:
        print("  - 5-step recursive prediction error is extremely small (< 10^-4).")
        print("  - The learned operators are highly suited for real-time Model Predictive Control (MPC).")
    else:
        print(f"  - 5-step recursive potential error is: {potential_act_errors[4]:.6e}")
        
    if not args.no_plot:
        if 'seaborn-v0_8-whitegrid' in plt.style.available:
            plt.style.use('seaborn-v0_8-whitegrid')
        else:
            plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        steps = np.arange(1, 6)
        ax1.plot(steps, subspace_errors, 'o-', color='#1f77b4', linewidth=2.0, markersize=8)
        ax1.set_yscale('log')
        ax1.set_title("Subspace Coordinate Prediction Error", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Horizon Steps Ahead (n)", fontsize=11)
        ax1.set_ylabel("Subspace State MSE (log scale)", fontsize=11)
        ax1.set_xticks(steps)
        ax1.grid(True, which="both", linestyle='--', alpha=0.6)
        
        ax2.plot(steps, potential_act_errors, 's--', color='#d62728', linewidth=2.0, markersize=8)
        ax2.set_yscale('log')
        ax2.set_title("Membrane Potential v Prediction Error (vs Actual)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Horizon Steps Ahead (n)", fontsize=11)
        ax2.set_ylabel("Membrane Potential MSE (log scale)", fontsize=11)
        ax2.set_xticks(steps)
        ax2.grid(True, which="both", linestyle='--', alpha=0.6)
        
        plt.suptitle(f"Bilinear DMDc Operator Short-Horizon Error Scaling\n(Evaluated on {args.test_batch} randomized test trajectories)", 
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, 'fhn_short_horizon_errors.png')
        plt.savefig(save_path, dpi=300)
        print(f"\nSaved Error scaling visualization plot to {save_path}")
        plt.close()

if __name__ == '__main__':
    main()
