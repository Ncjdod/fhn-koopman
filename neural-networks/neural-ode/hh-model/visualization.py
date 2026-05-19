"""
Visualization & Plotting for HH Neural ODE Training

Provides progress and final plotting functions used during
and after training. Separated from train.py for modularity.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from HH_NeuralODE import integrate
from AllenBrainLoader import SPECIMEN_ID


def plot_progress(model, hh, t_data, v_data, c_data, I_ext_fn,
                  epoch, info, loss_history, n_segments=None,
                  save_dir="HH_model"):
    """Plot current model predictions vs data with shooting structure."""
    os.makedirs(save_dir, exist_ok=True)

    y0 = hh.resting_state(v_data[0])
    y_pred = integrate(model, y0, t_data, I_ext_fn)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Voltage trace ---
    ax = axes[0, 0]
    ax.plot(t_data, v_data, 'b-', lw=1.5, label='Allen Data', alpha=0.7)
    ax.plot(t_data, y_pred[:, 0], 'r--', lw=1.5, label='Neural ODE')

    # Show shooting segment boundaries
    if n_segments is not None and n_segments > 1:
        boundaries = np.linspace(float(t_data[0]), float(t_data[-1]), n_segments + 1)
        for b in boundaries[1:-1]:
            ax.axvline(b, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(boundaries[1], color='gray', linestyle=':', alpha=0.4,
                   label=f'{n_segments} segments')

    ax.set_ylabel('Voltage (mV)')
    ax.set_title(f'Epoch {epoch}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Stimulus ---
    ax = axes[0, 1]
    ax.plot(t_data, c_data, 'g-', lw=1.5)
    ax.set_ylabel('Current (pA)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Stimulus')
    ax.grid(True, alpha=0.3)

    # --- Loss history ---
    ax = axes[1, 0]
    if loss_history:
        epochs_h = [h['epoch'] for h in loss_history]
        data_h = [h['data_loss'] for h in loss_history]
        phys_h = [h['physics_loss'] for h in loss_history]
        cont_h = [h.get('continuity_loss', 0.0) for h in loss_history]
        ax.semilogy(epochs_h, data_h, 'b-', label='Data Loss', alpha=0.7)
        ax.semilogy(epochs_h, phys_h, 'r-', label='Physics Loss', alpha=0.7)
        if any(c > 0 for c in cont_h):
            ax.semilogy(epochs_h, cont_h, 'g-', label='Continuity Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training History')
        ax.grid(True, alpha=0.3)

    # --- Weight evolution ---
    ax = axes[1, 1]
    if loss_history:
        mean_w = [h.get('mean_weight', 1.0) for h in loss_history]
        max_w = [h.get('max_weight', 1.0) for h in loss_history]
        ax.plot(epochs_h, mean_w, 'k-', label='Mean weight')
        ax.plot(epochs_h, max_w, 'r--', label='Max weight')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Adversarial Weight')
        ax.legend()
        ax.set_title('Loss Weight Evolution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_epoch_{epoch:05d}.png'), dpi=100)
    plt.close()


def plot_final(model, hh, t_data, v_data, c_data, I_ext_fn,
               loss_history, save_dir="HH_model"):
    """Final publication-quality plot."""
    os.makedirs(save_dir, exist_ok=True)

    y0 = hh.resting_state(v_data[0])
    y_pred = integrate(model, y0, t_data, I_ext_fn)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12),
                              gridspec_kw={'height_ratios': [3, 1, 2]})

    # --- Voltage ---
    ax = axes[0]
    ax.plot(t_data, v_data, 'b-', lw=2, label='Allen Brain Data', alpha=0.7)
    ax.plot(t_data, y_pred[:, 0], 'r--', lw=2, label='Neural ODE Prediction')
    ax.set_ylabel('Voltage (mV)', fontsize=12)
    ax.set_title(f'HH Neural ODE - Specimen {SPECIMEN_ID}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Stimulus ---
    ax = axes[1]
    ax.plot(t_data, c_data, 'g-', lw=2)
    ax.set_ylabel('I_ext (pA)', fontsize=12)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Loss curves ---
    ax = axes[2]
    if loss_history:
        epochs_h = [h['epoch'] for h in loss_history]
        data_h = [h['data_loss'] for h in loss_history]
        phys_h = [h['physics_loss'] for h in loss_history]
        total_h = [h['total_loss'] for h in loss_history]
        cont_h = [h.get('continuity_loss', 0.0) for h in loss_history]
        ax.semilogy(epochs_h, total_h, 'k-', lw=2, label='Total', alpha=0.8)
        ax.semilogy(epochs_h, data_h, 'b-', lw=1.5, label='Data', alpha=0.6)
        ax.semilogy(epochs_h, phys_h, 'r-', lw=1.5, label='Physics', alpha=0.6)
        if any(c > 0 for c in cont_h):
            ax.semilogy(epochs_h, cont_h, 'g-', lw=1.5, label='Continuity', alpha=0.6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_result.png'), dpi=150)
    plt.close()
    print(f"Saved final_result.png")
