"""
Visualization Utilities for HH Vector Field Learning

Provides plotting functions for both training phases:
  Phase 1: derivative scatter plots, phase portraits, training curves
  Phase 2: boundary fit, latent gating variables, integration test
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


# ================================================================
# Phase 1 Plots
# ================================================================

def plot_derivative_scatter(model, hh, sampler, key, n_samples=5000,
                            epoch=0, save_dir="HH_Field_Model"):
    """
    Scatter plot: predicted vs true derivatives for all 4 components.

    Each subplot shows one component (dV, dm, dh, dn) with the
    identity line y=x. Points clustering along this line = good fit.
    """
    os.makedirs(save_dir, exist_ok=True)

    states, I_ext = sampler.mixed_sample(key, n_samples)
    dydt_true = hh.derivatives_batch(states, I_ext)
    dydt_pred = model.predict_batch(states, I_ext)

    labels = ['dV/dt (mV/ms)', 'dm/dt (1/ms)', 'dh/dt (1/ms)', 'dn/dt (1/ms)']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (ax, label) in enumerate(zip(axes, labels)):
        true_i = np.array(dydt_true[:, i])
        pred_i = np.array(dydt_pred[:, i])

        ax.scatter(true_i, pred_i, s=1, alpha=0.3, c='steelblue')

        # Identity line
        lo = min(true_i.min(), pred_i.min())
        hi = max(true_i.max(), pred_i.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, alpha=0.7)

        r2 = 1.0 - np.var(pred_i - true_i) / max(np.var(true_i), 1e-12)
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(f'{label}\nR² = {r2:.4f}')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Vector Field Quality — Epoch {epoch}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'field_scatter_epoch_{epoch:05d}.png'),
                dpi=120, bbox_inches='tight')
    plt.close()


def plot_phase_portrait(model, hh, I_ext_val=10.0, n_grid=25,
                        epoch=0, save_dir="HH_Field_Model"):
    """
    Phase plane portraits: V-m, V-h, V-n with vector field arrows.

    Shows both the true HH field (blue) and learned NN field (red)
    at fixed I_ext, with other gating variables at steady-state.
    """
    os.makedirs(save_dir, exist_ok=True)

    V_range = jnp.linspace(-100.0, 60.0, n_grid)
    gate_range = jnp.linspace(0.01, 0.99, n_grid)

    pairs = [
        ('m', 1, 'V vs m'),
        ('h', 2, 'V vs h'),
        ('n', 3, 'V vs n'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (gate_name, gate_idx, title) in zip(axes, pairs):
        VV, GG = jnp.meshgrid(V_range, gate_range)
        VV_flat = VV.ravel()
        GG_flat = GG.ravel()

        # Build full states: fix other gates at steady-state of V
        m_vals = jnp.where(gate_idx == 1, GG_flat, hh.m_inf(VV_flat))
        h_vals = jnp.where(gate_idx == 2, GG_flat, hh.h_inf(VV_flat))
        n_vals = jnp.where(gate_idx == 3, GG_flat, hh.n_inf(VV_flat))

        states = jnp.stack([VV_flat, m_vals, h_vals, n_vals], axis=-1)
        I_ext = jnp.full(len(VV_flat), I_ext_val)

        # True field
        dydt_true = hh.derivatives_batch(states, I_ext)
        dV_true = np.array(dydt_true[:, 0]).reshape(n_grid, n_grid)
        dg_true = np.array(dydt_true[:, gate_idx]).reshape(n_grid, n_grid)

        # Predicted field
        dydt_pred = model.predict_batch(states, I_ext)
        dV_pred = np.array(dydt_pred[:, 0]).reshape(n_grid, n_grid)
        dg_pred = np.array(dydt_pred[:, gate_idx]).reshape(n_grid, n_grid)

        VV_np = np.array(VV)
        GG_np = np.array(GG)

        ax.quiver(VV_np, GG_np, dV_true, dg_true,
                  color='steelblue', alpha=0.5, label='HH (true)')
        ax.quiver(VV_np, GG_np, dV_pred, dg_pred,
                  color='tomato', alpha=0.5, label='NN (learned)')

        ax.set_xlabel('V (mV)')
        ax.set_ylabel(gate_name)
        ax.set_title(f'{title} (I_ext={I_ext_val})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Phase Portraits — Epoch {epoch}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'phase_portrait_epoch_{epoch:05d}.png'),
                dpi=120, bbox_inches='tight')
    plt.close()


def plot_training_curves(loss_history, save_dir="HH_Field_Model"):
    """Plot Phase 1 training loss curves with validation and integration metrics."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = [h['epoch'] for h in loss_history]
    total = [h['field_loss'] for h in loss_history]
    mse_dV = [h['mse_dV'] for h in loss_history]
    mse_dm = [h['mse_dm'] for h in loss_history]
    mse_dh = [h['mse_dh'] for h in loss_history]
    mse_dn = [h['mse_dn'] for h in loss_history]

    has_val = any('val_field_loss' in h for h in loss_history)
    has_int = any('int_v_mse' in h for h in loss_history)
    n_plots = 2 + (1 if has_val else 0) + (1 if has_int else 0)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    # Train + val normalized loss
    ax = axes[ax_idx]; ax_idx += 1
    ax.semilogy(epochs, total, 'k-', lw=2, label='Train')
    if has_val:
        val_epochs = [h['epoch'] for h in loss_history if 'val_field_loss' in h]
        val_loss = [h['val_field_loss'] for h in loss_history if 'val_field_loss' in h]
        ax.semilogy(val_epochs, val_loss, 'ro-', markersize=4, lw=1.5, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Field Loss')
    ax.set_title('Phase 1: Field Distillation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-component raw MSE
    ax = axes[ax_idx]; ax_idx += 1
    ax.semilogy(epochs, mse_dV, 'b-', label='dV/dt', alpha=0.8)
    ax.semilogy(epochs, mse_dm, 'r-', label='dm/dt', alpha=0.8)
    ax.semilogy(epochs, mse_dh, 'g-', label='dh/dt', alpha=0.8)
    ax.semilogy(epochs, mse_dn, 'm-', label='dn/dt', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Raw MSE')
    ax.set_title('Per-Component MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Integration validation
    if has_int:
        ax = axes[ax_idx]; ax_idx += 1
        int_epochs = [h['epoch'] for h in loss_history if 'int_v_mse' in h]
        int_mse = [h['int_v_mse'] for h in loss_history if 'int_v_mse' in h]
        int_max = [h['int_v_max_err'] for h in loss_history if 'int_v_max_err' in h]
        ax.plot(int_epochs, int_mse, 'b-o', markersize=4, lw=1.5, label='V MSE (mV²)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Integration V MSE (mV²)')
        ax.set_title('Integration Test')
        ax2 = ax.twinx()
        ax2.plot(int_epochs, int_max, 'r--s', markersize=4, lw=1.5, label='V max err (mV)')
        ax2.set_ylabel('Max |V error| (mV)', color='r')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase1_training_curves.png'), dpi=120)
    plt.close()


def plot_integration_test(model, hh, I_ext_val=10.0, T_ms=50.0,
                          dt=0.01, save_dir="HH_Field_Model"):
    """
    Forward-integrate both the learned NN and true HH from the same IC,
    overlay the voltage traces. This is the ultimate Phase 1 validation.

    Uses simple Euler integration (no diffrax dependency).
    """
    os.makedirs(save_dir, exist_ok=True)

    y0 = hh.resting_state(-65.0)
    n_steps = int(T_ms / dt)
    t = jnp.linspace(0.0, T_ms, n_steps)

    # Euler integration (HH ground truth)
    ys_hh = [y0]
    y = y0
    for i in range(n_steps - 1):
        dydt = hh._derivatives_single(y, I_ext_val)
        y = y + dt * dydt
        ys_hh.append(y)
    ys_hh = jnp.stack(ys_hh)

    # Euler integration (NN)
    ys_nn = [y0]
    y = y0
    for i in range(n_steps - 1):
        dydt = model(y[0], y[1], y[2], y[3], I_ext_val)
        y = y + dt * dydt
        ys_nn.append(y)
    ys_nn = jnp.stack(ys_nn)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, ys_hh[:, 0], 'b-', lw=2, label='HH (true)')
    ax.plot(t, ys_nn[:, 0], 'r--', lw=2, label='NN (learned)')
    ax.set_ylabel('V (mV)')
    ax.set_title(f'Integration Test: I_ext = {I_ext_val} uA/cm², {T_ms}ms')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, (label, color) in enumerate(zip(['m', 'h', 'n'], ['orange', 'green', 'purple'])):
        ax.plot(t, ys_hh[:, i+1], '-', color=color, lw=1.5, alpha=0.7,
                label=f'{label} (HH)')
        ax.plot(t, ys_nn[:, i+1], '--', color=color, lw=1.5, alpha=0.7,
                label=f'{label} (NN)')
    ax.set_ylabel('Gating')
    ax.set_xlabel('Time (ms)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'integration_test.png'), dpi=120)
    plt.close()


# ================================================================
# Phase 2 Plots
# ================================================================

def plot_boundary_fit(model, V_obs, I_ext_hh, latent_gates, dVdt_obs,
                      t_ms, epoch=0, save_dir="HH_Field_Model"):
    """
    Phase 2 boundary condition fit: predicted vs observed dV/dt,
    and the learned latent gating variables.
    """
    os.makedirs(save_dir, exist_ok=True)

    m_lat = np.array(latent_gates.m)
    h_lat = np.array(latent_gates.h)
    n_lat = np.array(latent_gates.n)
    t_np = np.array(t_ms)
    V_np = np.array(V_obs)
    dVdt_np = np.array(dVdt_obs)

    # Predict along trajectory
    states = jnp.stack([V_obs, latent_gates.m, latent_gates.h, latent_gates.n],
                       axis=-1)
    dydt_pred = model.predict_batch(states, I_ext_hh)
    dVdt_pred = np.array(dydt_pred[:, 0])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Voltage trace
    ax = axes[0]
    ax.plot(t_np, V_np, 'b-', lw=2, label='Allen Data V(t)')
    ax.set_ylabel('V (mV)')
    ax.set_title(f'Phase 2 Boundary Fit — Epoch {epoch}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # dV/dt comparison
    ax = axes[1]
    ax.plot(t_np, dVdt_np, 'b-', lw=1.5, alpha=0.7, label='Observed dV/dt')
    ax.plot(t_np, dVdt_pred, 'r--', lw=1.5, alpha=0.7, label='Predicted dV/dt')
    ax.set_ylabel('dV/dt (mV/ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Latent gating variables
    ax = axes[2]
    ax.plot(t_np, m_lat, 'orange', lw=1.5, label='m (latent)')
    ax.plot(t_np, h_lat, 'green', lw=1.5, label='h (latent)')
    ax.plot(t_np, n_lat, 'purple', lw=1.5, label='n (latent)')

    # Overlay HH steady-state for comparison
    from hh_reference import HHReference
    hh = HHReference()
    ax.plot(t_np, np.array(hh.m_inf(V_obs)), ':', color='orange', alpha=0.5,
            label='m_inf(V)')
    ax.plot(t_np, np.array(hh.h_inf(V_obs)), ':', color='green', alpha=0.5,
            label='h_inf(V)')
    ax.plot(t_np, np.array(hh.n_inf(V_obs)), ':', color='purple', alpha=0.5,
            label='n_inf(V)')

    ax.set_ylabel('Gating Variable')
    ax.set_xlabel('Time (ms)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'boundary_fit_epoch_{epoch:05d}.png'),
                dpi=120)
    plt.close()


def plot_phase2_curves(loss_history, save_dir="HH_Field_Model"):
    """Plot Phase 2 training loss components."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = [h['epoch'] for h in loss_history]
    total = [h['total_loss'] for h in loss_history]
    dV = [h['dV_loss'] for h in loss_history]
    gate = [h['gate_loss'] for h in loss_history]
    smooth = [h['smooth_loss'] for h in loss_history]
    field = [h['field_loss'] for h in loss_history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, total, 'k-', lw=2, label='Total')
    ax.semilogy(epochs, dV, 'b-', lw=1.5, alpha=0.7, label='dV/dt')
    ax.semilogy(epochs, gate, 'r-', lw=1.5, alpha=0.7, label='Gating')
    ax.semilogy(epochs, smooth, 'g-', lw=1.5, alpha=0.7, label='Smoothness')
    ax.semilogy(epochs, field, 'm-', lw=1.5, alpha=0.7, label='Field (anti-forget)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Phase 2: Boundary Condition Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase2_training_curves.png'), dpi=120)
    plt.close()
