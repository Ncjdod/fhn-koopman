"""
Phase 1: Vector Field Distillation Training

Trains VectorFieldNet to approximate the full HH vector field by
regressing on exact derivatives at randomly sampled state-space points.

No path integration — just supervised learning on (state, derivative) pairs.
New batch generated online each epoch for maximum coverage.

Usage:
    python train_field.py
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from config import Phase1Config
from hh_reference import HHReference
from model import create_model
from sampler import StateSpaceSampler
from losses import field_loss
from visualization import (plot_derivative_scatter, plot_phase_portrait,
                           plot_training_curves, plot_integration_test)


def make_train_step(optimizer):
    """
    Create a JIT-compiled training step.

    Returns:
        step_fn(model, opt_state, states, I_ext, dydt_true) -> (model, opt_state, loss, info)
    """
    @eqx.filter_jit
    def step(model, opt_state, states, I_ext, dydt_true):
        (loss, info), grads = eqx.filter_value_and_grad(field_loss, has_aux=True)(
            model, states, I_ext, dydt_true
        )

        # Clean NaN/inf gradients
        grads = jax.tree.map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
        )

        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model_new = eqx.apply_updates(model, updates)

        return model_new, opt_state_new, loss, info

    return step


def train_phase1(config=None):
    """
    Run Phase 1: HH vector field distillation.

    Pipeline:
        1. Create model + optimizer
        2. For each epoch:
           a. Sample random (state, I_ext) from state space
           b. Compute true HH derivatives
           c. Compute field_loss + backprop
           d. Update model
        3. Periodically: scatter plots, phase portraits, integration test
        4. Save checkpoints

    Args:
        config: Phase1Config instance (default: Phase1Config())

    Returns:
        model:        Trained VectorFieldNet
        loss_history: List of info dicts
    """
    if config is None:
        config = Phase1Config()

    print("=" * 60)
    print("Phase 1: HH Vector Field Distillation")
    print("=" * 60)

    # ---- Setup ----
    key = jax.random.PRNGKey(config.seed)
    hh = HHReference()
    sampler = StateSpaceSampler(
        V_range=config.V_range,
        m_range=config.m_range,
        h_range=config.h_range,
        n_range=config.n_range,
        I_ext_range=config.I_ext_range,
        V_mean=config.V_mean,
        V_std=config.V_std,
        gate_std=config.gate_std,
    )

    # ---- Model ----
    key, model_key = jax.random.split(key)
    model = create_model(
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        key=model_key,
    )

    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}")

    # ---- Optimizer: AdamW with cosine schedule ----
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.lr,
        decay_steps=config.n_epochs,
        alpha=config.lr_min / config.lr,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=config.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ---- JIT-compiled train step ----
    step_fn = make_train_step(optimizer)

    # ---- Training loop ----
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    loss_history = []
    start_time = time.time()
    best_loss = float('inf')

    print(f"\nTraining: {config.n_epochs} epochs, batch_size={config.batch_size}")
    print(f"Sampling: {config.physiological_fraction*100:.0f}% physiological, "
          f"{(1-config.physiological_fraction)*100:.0f}% uniform")
    print()

    for epoch in range(config.n_epochs):
        # Generate fresh batch
        key, sample_key = jax.random.split(key)
        states, I_ext = sampler.mixed_sample(
            sample_key, config.batch_size, config.physiological_fraction
        )

        # Compute ground truth derivatives
        dydt_true = hh.derivatives_batch(states, I_ext)

        # Training step
        model, opt_state, loss, info = step_fn(
            model, opt_state, states, I_ext, dydt_true
        )

        # ---- Logging ----
        do_log = (epoch % config.log_every == 0)
        do_plot = (epoch % config.plot_every == 0) and epoch > 0
        do_val = (epoch % config.val_every == 0) and epoch > 0
        do_ckpt = (epoch % config.checkpoint_every == 0) and epoch > 0

        if do_log or do_plot or do_val or do_ckpt:
            info_np = {k: float(v) for k, v in info.items()}
            info_np['epoch'] = epoch
            info_np['lr'] = float(lr_schedule(epoch))
            loss_history.append(info_np)

            if do_log:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch:>5} | "
                      f"Loss: {info_np['field_loss']:.6f} | "
                      f"MSE dV: {info_np['mse_dV']:.4f} | "
                      f"MSE dm: {info_np['mse_dm']:.6f} | "
                      f"MSE dh: {info_np['mse_dh']:.6f} | "
                      f"MSE dn: {info_np['mse_dn']:.6f} | "
                      f"LR: {info_np['lr']:.2e} | "
                      f"{elapsed:.0f}s")

            if do_plot:
                key, plot_key = jax.random.split(key)
                save_dir = os.path.dirname(config.checkpoint_dir)
                plot_derivative_scatter(
                    model, hh, sampler, plot_key,
                    epoch=epoch, save_dir=save_dir
                )
                plot_phase_portrait(
                    model, hh, I_ext_val=10.0,
                    epoch=epoch, save_dir=save_dir
                )

            if do_val:
                save_dir = os.path.dirname(config.checkpoint_dir)
                plot_integration_test(
                    model, hh, I_ext_val=10.0, T_ms=50.0,
                    save_dir=save_dir
                )

            if do_ckpt:
                ckpt_path = os.path.join(
                    config.checkpoint_dir, f"phase1_epoch_{epoch:05d}.eqx"
                )
                eqx.tree_serialise_leaves(ckpt_path, model)

                if info_np['field_loss'] < best_loss:
                    best_loss = info_np['field_loss']
                    best_path = os.path.join(config.checkpoint_dir, "phase1_best.eqx")
                    eqx.tree_serialise_leaves(best_path, model)

    # ---- Final ----
    elapsed_total = time.time() - start_time
    print(f"\n--- Phase 1 Complete ({elapsed_total:.0f}s) ---")
    print(f"Final field loss: {loss_history[-1]['field_loss']:.6f}")
    print(f"Best field loss:  {best_loss:.6f}")

    # Final plots
    save_dir = os.path.dirname(config.checkpoint_dir)
    plot_training_curves(loss_history, save_dir=save_dir)

    key, plot_key = jax.random.split(key)
    plot_derivative_scatter(
        model, hh, sampler, plot_key,
        epoch=config.n_epochs, save_dir=save_dir
    )
    plot_phase_portrait(
        model, hh, I_ext_val=10.0,
        epoch=config.n_epochs, save_dir=save_dir
    )
    plot_integration_test(model, hh, I_ext_val=10.0, T_ms=50.0, save_dir=save_dir)

    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "phase1_final.eqx")
    eqx.tree_serialise_leaves(final_path, model)
    print(f"Model saved: {final_path}")

    return model, loss_history


if __name__ == "__main__":
    model, history = train_phase1()
