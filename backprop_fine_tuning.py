"""
Bilinear DMDc Backpropagation Fine-Tuning and Dynamic Chirp Validation in JAX.
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from dynamics import run_dmdc, get_external_current
from simulation import simulate_fhn, simulate_fhn_batch

def predict_trajectory(v_true, u_true, A_tilde, B_tilde, C_tilde, Ur, H):
    """Recursively reconstructs the potential trajectory using subspace operators."""
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

def predict_subspace_multistep(z_init, u_seq, A, B, C):
    """Predicts forward in the subspace for a short horizon."""
    def step(z, u):
        z_next = A @ z + B[:, 0] * u + C @ z * u
        return z_next, z_next
    _, z_preds = jax.lax.scan(step, z_init, u_seq)
    return z_preds

def main():
    """Simulates FHN data, performs standard DMDc, optimizes operators via BPTT, and validates on chirp current."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Bilinear DMDc Backprop Fine-Tuning")
    parser.add_argument('--batch-size', type=int, default=8, help="Number of trajectories for training")
    parser.add_argument('--dmd-H', type=int, default=70, help="Delay embedding dimension H")
    parser.add_argument('--dmd-r', type=int, default=12, help="Truncation rank r")
    parser.add_argument('--dmd-p', type=int, default=20, help="Truncation rank p")
    parser.add_argument('--t-max', type=float, default=100.0, help="Simulation time")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step")
    parser.add_argument('--lr', type=float, default=2e-3, help="Learning rate")
    parser.add_argument('--steps', type=int, default=150, help="Optimization steps")
    parser.add_argument('--n-predict', type=int, default=30, help="Prediction horizon for multiple-shooting")
    parser.add_argument('--stride', type=int, default=5, help="Slicing stride for multiple-shooting")
    parser.add_argument('--no-plot', action='store_true', help="Disable matplotlib plotting")
    args = parser.parse_args()
    
    n_steps = int(args.t_max / args.dt) + 1
    t_span = jnp.linspace(0.0, args.t_max, n_steps)
    
    key = jax.random.PRNGKey(101)
    key1, key2, key3 = jax.random.split(key, 3)
    v0s = jax.random.uniform(key1, (args.batch_size,), minval=-2.0, maxval=1.0)
    w0s = jax.random.uniform(key2, (args.batch_size,), minval=-1.0, maxval=0.5)
    y0_batch = jnp.stack([v0s, w0s], axis=1)
    I_val_batch = jax.random.uniform(key3, (args.batch_size,), minval=0.2, maxval=1.2)
    
    print(f"Simulating {args.batch_size} FHN trajectories in parallel...")
    ys = simulate_fhn_batch(y0_batch, t_span, 'sine', I_val_batch)
    
    u_data_batch = jnp.stack([
        jnp.array([get_external_current(t, 'sine', iv) for t in t_span])
        for iv in I_val_batch
    ], axis=0)
    
    print("Learning initial operators with standard Bilinear DMDc...")
    A_init, B_init, C_init, _, _, _, dmdc_X, _, dmdc_U, Ur = run_dmdc(
        ys[:, :, 0], u_data_batch, H=args.dmd_H, r=args.dmd_r, p=args.dmd_p
    )
    
    K = n_steps - args.dmd_H + 1
    X_global = dmdc_X
    Uc_global = dmdc_U
    
    X_split = jnp.split(X_global, args.batch_size, axis=1)
    Uc_split = jnp.split(Uc_global, args.batch_size, axis=1)
    
    z_inits_list = []
    u_seqs_list = []
    z_targets_list = []
    
    S = (K - 1 - args.n_predict) // args.stride
    
    for m in range(args.batch_size):
        X_m = X_split[m]
        Uc_m = Uc_split[m][0]
        Z_m = Ur.T @ X_m
        
        for j in range(S):
            k = j * args.stride
            z_inits_list.append(Z_m[:, k])
            u_seqs_list.append(Uc_m[k : k + args.n_predict])
            z_targets_list.append(Z_m[:, k + 1 : k + 1 + args.n_predict].T)
            
    z_inits = jnp.stack(z_inits_list, axis=0)
    u_seqs = jnp.stack(u_seqs_list, axis=0)
    z_targets = jnp.stack(z_targets_list, axis=0)
    
    params = (A_init, B_init, C_init)
    
    def loss_fn(p_vars, z_in, u_in, z_tgt):
        A, B, C = p_vars
        vmapped_loss = jax.vmap(
            lambda z_init, u_seq, z_target: jnp.mean((predict_subspace_multistep(z_init, u_seq, A, B, C) - z_target) ** 2),
            in_axes=(0, 0, 0)
        )
        return jnp.mean(vmapped_loss(z_in, u_in, z_tgt))
        
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(p_vars, state, z_in, u_in, z_tgt):
        loss, grads = jax.value_and_grad(loss_fn)(p_vars, z_in, u_in, z_tgt)
        updates, state = optimizer.update(grads, state, p_vars)
        p_vars = optax.apply_updates(p_vars, updates)
        return p_vars, state, loss
        
    initial_loss = float(loss_fn(params, z_inits, u_seqs, z_targets))
    print(f"\nInitial Multiple-Shooting Training Loss (MSE): {initial_loss:.6e}")
    
    print(f"Fine-tuning Bilinear DMDc operators using Backprop through Time ({args.steps} steps)...")
    for step in range(args.steps):
        params, opt_state, loss_val = train_step(params, opt_state, z_inits, u_seqs, z_targets)
        if step % 15 == 0 or step == args.steps - 1:
            print(f"Step {step:03d} | Subspace Loss (MSE): {float(loss_val):.6e}")
            
    final_loss = float(loss_fn(params, z_inits, u_seqs, z_targets))
    print(f"Optimized Multiple-Shooting Training Loss (MSE): {final_loss:.6e}")
    
    A_opt, B_opt, C_opt = params
    
    print("\nSimulating validation trajectory under completely unseen dynamic chirp current...")
    y0_val = jnp.array([-1.5, -0.5])
    I_val_chirp = 0.5
    ys_chirp_true = simulate_fhn(y0_val, t_span, I_type='chirp', I_val=I_val_chirp)
    u_chirp_true = jnp.array([get_external_current(t, 'chirp', I_val_chirp) for t in t_span])
    
    v_chirp_true = ys_chirp_true[:, 0]
    
    print("Testing recursive reconstruction with unoptimized operators...")
    v_chirp_pred_init = predict_trajectory(v_chirp_true, u_chirp_true, A_init, B_init, C_init, Ur, args.dmd_H)
    mse_init = float(np.mean((v_chirp_true - v_chirp_pred_init) ** 2))
    print(f"Unoptimized operators validation MSE: {mse_init:.6e}")
    
    print("Testing recursive reconstruction with optimized operators...")
    v_chirp_pred_opt = predict_trajectory(v_chirp_true, u_chirp_true, A_opt, B_opt, C_opt, Ur, args.dmd_H)
    mse_opt = float(np.mean((v_chirp_true - v_chirp_pred_opt) ** 2))
    print(f"Optimized operators validation MSE: {mse_opt:.6e}")
    
    improvement = (mse_init - mse_opt) / mse_init * 100
    print(f"Generalization Accuracy Improvement on Unseen Profile: {improvement:.2f}%")
    
    if not args.no_plot:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        ax1.plot(t_span, u_chirp_true, color='#d62728', linewidth=1.5, label='Dynamic Chirp Input Current')
        ax1.set_title("Unseen Dynamic Stimulus Current Profile (Frequency Sweep)", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Stimulus Magnitude", fontsize=11)
        ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2.plot(t_span, v_chirp_true, label='Ground Truth FHN Simulation', color='#1f77b4', linewidth=2.0)
        ax2.plot(t_span, v_chirp_pred_init, ':', label=f'Unoptimized Bilinear DMDc (MSE: {mse_init:.4f})', color='#ff7f0e', linewidth=1.5)
        ax2.plot(t_span, v_chirp_pred_opt, '--', label=f'BPTT Optimized Bilinear DMDc (MSE: {mse_opt:.4f})', color='#2ca02c', linewidth=2.0)
        ax2.set_title("Bilinear DMDc Out-of-Distribution Validation (True vs Reconstructed)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Time (dimensionless)", fontsize=11)
        ax2.set_ylabel("Membrane Potential v", fontsize=11)
        ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        save_path = os.path.join(plots_dir, 'fhn_backprop_verification.png')
        plt.savefig(save_path, dpi=300)
        print(f"\nSaved Comparative verification plot to {save_path}")
        plt.close()

if __name__ == '__main__':
    main()
