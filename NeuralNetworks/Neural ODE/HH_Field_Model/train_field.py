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


# ================================================================
# Validation: fixed field set + integration trajectory
# ================================================================

def make_val_field_fn():
    """JIT-compiled field loss on a fixed validation set (no gradients)."""
    @eqx.filter_jit
    def val_field(model, val_states, val_I, val_dydt, sigma):
        _, info = field_loss(model, val_states, val_I, val_dydt, sigma)
        return info
    return val_field


def _make_integration_mse_fn():
    """
    Create a JIT-compiled integration test via jax.lax.scan.

    Much faster than a Python for-loop (~100x speedup).
    """
    @eqx.filter_jit
    def _integration_mse(model, hh_y0, I_ext_val, n_steps, dt):
        # Euler via lax.scan: HH ground truth
        def hh_step(y, _):
            I_Na = 120.0 * (y[1]**3) * y[2] * (y[0] - 50.0)
            I_K = 36.0 * (y[3]**4) * (y[0] - (-77.0))
            I_L = 0.3 * (y[0] - (-54.4))
            dV = (I_ext_val - I_Na - I_K - I_L) / 1.0

            # alpha/beta with safe division
            dv_m = y[0] + 40.0
            safe_m = jnp.where(jnp.abs(dv_m) < 1e-6, 1.0, dv_m)
            a_m = jnp.where(jnp.abs(dv_m) < 1e-6, 1.0,
                            0.1 * safe_m / (1.0 - jnp.exp(-safe_m / 10.0)))
            b_m = 4.0 * jnp.exp(-(y[0] + 65.0) / 18.0)

            a_h = 0.07 * jnp.exp(-(y[0] + 65.0) / 20.0)
            b_h = 1.0 / (1.0 + jnp.exp(-(y[0] + 35.0) / 10.0))

            dv_n = y[0] + 55.0
            safe_n = jnp.where(jnp.abs(dv_n) < 1e-6, 1.0, dv_n)
            a_n = jnp.where(jnp.abs(dv_n) < 1e-6, 0.1,
                            0.01 * safe_n / (1.0 - jnp.exp(-safe_n / 10.0)))
            b_n = 0.125 * jnp.exp(-(y[0] + 65.0) / 80.0)

            dm = a_m * (1.0 - y[1]) - b_m * y[1]
            dh = a_h * (1.0 - y[2]) - b_h * y[2]
            dn = a_n * (1.0 - y[3]) - b_n * y[3]

            dy = jnp.array([dV, dm, dh, dn])
            y_new = y + dt * dy
            return y_new, y_new[0]

        _, V_hh = jax.lax.scan(hh_step, hh_y0, None, length=n_steps)

        # Euler via lax.scan: NN prediction (with gate clipping for safety)
        def nn_step(y, _):
            dy = model(y[0], y[1], y[2], y[3], I_ext_val)
            y_new = y + dt * dy
            # Clip gates to [0,1] to match evaluation-time behavior
            y_new = y_new.at[1].set(jnp.clip(y_new[1], 0.0, 1.0))
            y_new = y_new.at[2].set(jnp.clip(y_new[2], 0.0, 1.0))
            y_new = y_new.at[3].set(jnp.clip(y_new[3], 0.0, 1.0))
            return y_new, y_new[0]

        _, V_nn = jax.lax.scan(nn_step, hh_y0, None, length=n_steps)

        v_mse = jnp.mean((V_hh - V_nn) ** 2)
        v_max_err = jnp.max(jnp.abs(V_hh - V_nn))

        return v_mse, v_max_err

    return _integration_mse


def integration_mse(model, hh, I_ext_val=10.0, T_ms=50.0, dt=0.01,
                    _cached_fn=[None]):
    """
    Forward-integrate learned field vs HH ground truth, return voltage MSE.

    Uses Euler via jax.lax.scan — JIT compiled, ~100x faster than Python loop.
    """
    if _cached_fn[0] is None:
        _cached_fn[0] = _make_integration_mse_fn()

    y0 = hh.resting_state(-65.0)
    n_steps = int(T_ms / dt)

    v_mse, v_max_err = _cached_fn[0](model, y0, I_ext_val, n_steps, dt)

    return float(v_mse), float(v_max_err)


def make_train_step(optimizer):
    """
    Create a JIT-compiled training step.

    Returns:
        step_fn(model, opt_state, states, I_ext, dydt_true) -> (model, opt_state, loss, info)
    """
    @eqx.filter_jit
    def step(model, opt_state, states, I_ext, dydt_true, sigma):
        (loss, info), grads = eqx.filter_value_and_grad(field_loss, has_aux=True)(
            model, states, I_ext, dydt_true, sigma
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

    # ---- Global sigma: frozen normalization from large sample ----
    # sigma[0] is computed on log-transformed dV (matching field_loss transform)
    from losses import _log_transform
    key, sigma_key = jax.random.split(key)
    sigma_states, sigma_I = sampler.mixed_sample(sigma_key, 50_000, config.physiological_fraction)
    sigma_dydt = hh.derivatives_batch(sigma_states, sigma_I)
    dV_log_sigma = jnp.std(_log_transform(sigma_dydt[:, 0]))
    global_sigma = jnp.array([
        dV_log_sigma,
        jnp.std(sigma_dydt[:, 1]),
        jnp.std(sigma_dydt[:, 2]),
        jnp.std(sigma_dydt[:, 3]),
    ])
    global_sigma = jnp.maximum(global_sigma, 1e-6)
    print(f"Global sigma (log_dV, dm, dh, dn): "
          f"{float(global_sigma[0]):.4f}, {float(global_sigma[1]):.4f}, "
          f"{float(global_sigma[2]):.4f}, {float(global_sigma[3]):.4f}")

    # ---- Model ----
    key, model_key = jax.random.split(key)
    model = create_model(
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_fourier=getattr(config, 'n_fourier', 128),
        sigma=getattr(config, 'fourier_sigma', 10.0),
        head_dim=getattr(config, 'head_dim', 32),
        v_head_dim=getattr(config, 'v_head_dim', 64),
        key=model_key,
    )

    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}")

    # ---- Optimizer: AdamW with cosine schedule ----
    inner_steps = getattr(config, 'inner_steps', 1)
    total_steps = config.n_epochs * inner_steps
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.lr,
        decay_steps=total_steps,
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

    # ---- Fixed validation set (sampled once, reused every val_every) ----
    key, val_key = jax.random.split(key)
    val_states, val_I = sampler.mixed_sample(val_key, config.batch_size)
    val_dydt = hh.derivatives_batch(val_states, val_I)
    val_field_fn = make_val_field_fn()
    print(f"Validation set: {config.batch_size} fixed points")

    # ---- Device check ----
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    if jax.default_backend() != 'gpu':
        print("WARNING: Not using GPU! Install jax[cuda12] for CUDA support.")

    # ---- Warmup: small batch to trigger JIT compilation ----
    print(f"\nCompiling train step (one-time cost)...", flush=True)
    t_compile = time.time()

    key, warmup_key = jax.random.split(key)
    warmup_states, warmup_I = sampler.mixed_sample(warmup_key, 64, config.physiological_fraction)
    warmup_dydt = hh.derivatives_batch(warmup_states, warmup_I)
    _ = step_fn(model, opt_state, warmup_states, warmup_I, warmup_dydt, global_sigma)

    # Force computation to complete (JAX is lazy)
    jax.block_until_ready(_)

    print(f"Compilation done in {time.time() - t_compile:.1f}s")

    # ---- Training loop ----
    print(f"\nTraining: {config.n_epochs} epochs × {inner_steps} inner steps "
          f"= {total_steps} gradient steps, batch_size={config.batch_size}")
    print(f"Sampling: {config.physiological_fraction*100:.0f}% physiological, "
          f"{(1-config.physiological_fraction)*100:.0f}% uniform")
    print(flush=True)

    step_count = 0  # tracks total gradient steps for LR schedule

    for epoch in range(config.n_epochs):
        # Generate fresh batch (reused for inner_steps gradient steps)
        key, sample_key = jax.random.split(key)
        states, I_ext = sampler.mixed_sample(
            sample_key, config.batch_size, config.physiological_fraction
        )

        # Compute ground truth derivatives (once per batch)
        dydt_true = hh.derivatives_batch(states, I_ext)

        # Multiple gradient steps on the same batch
        for _ in range(inner_steps):
            model, opt_state, loss, info = step_fn(
                model, opt_state, states, I_ext, dydt_true, global_sigma
            )
            step_count += 1

        # ---- Logging ----
        # Epoch 0: always log to confirm training started
        do_log = (epoch % config.log_every == 0) or epoch == 0
        do_plot = (epoch % config.plot_every == 0) and epoch > 0
        do_val = (epoch % config.val_every == 0) and epoch > 0
        do_ckpt = (epoch % config.checkpoint_every == 0) and epoch > 0

        if do_log or do_plot or do_val or do_ckpt:
            info_np = {k: float(v) for k, v in info.items()}
            info_np['epoch'] = epoch
            info_np['lr'] = float(lr_schedule(step_count))
            loss_history.append(info_np)

            # Compute validation losses when needed
            if do_val:
                val_info = val_field_fn(model, val_states, val_I, val_dydt, global_sigma)
                info_np['val_field_loss'] = float(val_info['field_loss'])
                info_np['val_nmse_dV'] = float(val_info['nmse_dV'])
                info_np['val_nmse_dm'] = float(val_info['nmse_dm'])

                v_mse, v_max = integration_mse(model, hh, I_ext_val=10.0, T_ms=50.0)
                info_np['int_v_mse'] = v_mse
                info_np['int_v_max_err'] = v_max

            if do_log:
                elapsed = time.time() - start_time
                val_str = ""
                if 'val_field_loss' in info_np:
                    val_str = (f" | Val: {info_np['val_field_loss']:.6f}"
                               f" | IntMSE: {info_np['int_v_mse']:.2f}"
                               f" | IntMax: {info_np['int_v_max_err']:.1f}mV")
                print(f"  Epoch {epoch:>5} ({step_count:>6} steps) | "
                      f"Loss: {info_np['field_loss']:.6f} | "
                      f"nMSE dV: {info_np['nmse_dV']:.6f} | "
                      f"nMSE dm: {info_np['nmse_dm']:.6f} | "
                      f"nMSE dh: {info_np['nmse_dh']:.6f} | "
                      f"nMSE dn: {info_np['nmse_dn']:.6f} | "
                      f"LR: {info_np['lr']:.2e}"
                      f"{val_str} | "
                      f"{elapsed:.0f}s", flush=True)

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

    # ---- Final validation ----
    val_info_final = val_field_fn(model, val_states, val_I, val_dydt, global_sigma)
    v_mse_final, v_max_final = integration_mse(model, hh, I_ext_val=10.0, T_ms=50.0)

    elapsed_total = time.time() - start_time
    print(f"\n--- Phase 1 Complete ({elapsed_total:.0f}s) ---")
    print(f"Final train loss:  {loss_history[-1]['field_loss']:.6f}")
    print(f"Final val loss:    {float(val_info_final['field_loss']):.6f}")
    print(f"Integration V MSE: {v_mse_final:.2f} mV²")
    print(f"Integration V max: {v_max_final:.1f} mV")
    print(f"Best train loss:   {best_loss:.6f}")

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
