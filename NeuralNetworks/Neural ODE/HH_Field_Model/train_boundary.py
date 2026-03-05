"""
Phase 2: Allen Brain Boundary Condition Fine-Tuning

Jointly optimizes:
  - VectorFieldNet weights (small LR — preserve learned field)
  - Latent gating variables m(t), h(t), n(t) (larger LR)
  - Unit conversion factor pA -> uA/cm^2 (medium LR)

The boundary loss enforces consistency between the learned vector field
and the observed Allen Brain trajectory. An anti-forgetting field loss
on fresh HH samples prevents catastrophic forgetting.

Usage:
    python train_boundary.py                   # loads Phase 1 checkpoint
    python train_boundary.py --from_scratch     # trains Phase 1 first
"""

import os
import sys
import time
import argparse

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from config import Phase1Config, Phase2Config
from hh_reference import HHReference
from model import create_model, VectorFieldNet
from sampler import StateSpaceSampler
from data import load_allen_data, compute_dVdt
from latent_state import LatentGatingState, ConversionFactor
from losses import total_phase2_loss
from visualization import plot_boundary_fit, plot_phase2_curves


def _load_phase1_model(config_p1):
    """Load the best Phase 1 checkpoint."""
    best_path = os.path.join(config_p1.checkpoint_dir, "phase1_best.eqx")
    final_path = os.path.join(config_p1.checkpoint_dir, "phase1_final.eqx")

    path = best_path if os.path.exists(best_path) else final_path
    if not os.path.exists(path):
        return None

    # Create skeleton model to deserialize into
    key = jax.random.PRNGKey(0)
    skeleton = create_model(
        hidden_dim=config_p1.hidden_dim,
        n_layers=config_p1.n_layers,
        key=key,
    )
    model = eqx.tree_deserialise_leaves(path, skeleton)
    print(f"Loaded Phase 1 model: {path}")
    return model


def make_train_step(model_optimizer, latent_optimizer, conv_optimizer,
                    hh, sampler,
                    V_obs, I_ext_pA, dVdt_obs, t_ms,
                    field_weight, gating_weight, smooth_weight,
                    field_batch_size):
    """
    Create JIT-compiled Phase 2 training step.

    Jointly updates model, latent_gates, and conversion_factor.
    Uses separate optimizers with different learning rates for each.
    """

    @eqx.filter_jit
    def step(model, latent_gates, conversion_factor,
             model_opt_state, latent_opt_state, conv_opt_state,
             sampler_key):

        # Combined loss over all trainable parameters
        # eqx.filter_value_and_grad only differentiates the first arg,
        # so we pack all trainables into a single tuple.
        def loss_fn(params_tuple):
            model_, latent_, conv_ = params_tuple
            return total_phase2_loss(
                model_, latent_, conv_,
                V_obs, I_ext_pA, dVdt_obs, t_ms,
                hh, sampler, sampler_key,
                field_weight=field_weight,
                gating_weight=gating_weight,
                smooth_weight=smooth_weight,
                field_batch_size=field_batch_size,
            )

        # Compute gradients w.r.t. all three parameter groups (packed as tuple)
        (loss, info), grads_tuple = \
            eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                (model, latent_gates, conversion_factor)
            )
        grads_model, grads_latent, grads_conv = grads_tuple

        # Clean NaN/inf gradients
        def clean(g):
            return jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), g)

        grads_model = clean(grads_model)
        grads_latent = clean(grads_latent)
        grads_conv = clean(grads_conv)

        # Update model
        updates_m, model_opt_state_new = model_optimizer.update(
            grads_model, model_opt_state, model
        )
        model_new = eqx.apply_updates(model, updates_m)

        # Update latent gating
        updates_l, latent_opt_state_new = latent_optimizer.update(
            grads_latent, latent_opt_state, latent_gates
        )
        latent_new = eqx.apply_updates(latent_gates, updates_l)

        # Update conversion factor
        updates_c, conv_opt_state_new = conv_optimizer.update(
            grads_conv, conv_opt_state, conversion_factor
        )
        conv_new = eqx.apply_updates(conversion_factor, updates_c)

        # Clamp conversion factor to reasonable range
        # log(0.001) ~ -6.9, log(1.0) ~ 0.0 (membrane area 1e-3 to 1e-6 cm^2)
        conv_new = eqx.tree_at(
            lambda c: c.log_factor,
            conv_new,
            jnp.clip(conv_new.log_factor, -7.0, 1.0)
        )

        return (model_new, latent_new, conv_new,
                model_opt_state_new, latent_opt_state_new, conv_opt_state_new,
                info)

    return step


def train_phase2(model=None, config_p1=None, config_p2=None):
    """
    Run Phase 2: Fine-tune model with Allen Brain boundary condition.

    Args:
        model:     Pre-trained VectorFieldNet from Phase 1 (or None to load checkpoint)
        config_p1: Phase1Config (for loading checkpoint)
        config_p2: Phase2Config

    Returns:
        model:         Fine-tuned VectorFieldNet
        latent_gates:  Learned LatentGatingState
        loss_history:  List of info dicts
    """
    if config_p1 is None:
        config_p1 = Phase1Config()
    if config_p2 is None:
        config_p2 = Phase2Config()

    print("=" * 60, flush=True)
    print("Phase 2: Allen Brain Boundary Condition Fine-Tuning", flush=True)
    print("=" * 60, flush=True)

    # ---- Load Phase 1 model ----
    if model is None:
        model = _load_phase1_model(config_p1)
        if model is None:
            print("ERROR: No Phase 1 checkpoint found. Run train_field.py first.")
            return None, None, []

    # ---- Load Allen data ----
    t_np, V_np, I_np = load_allen_data(
        downsample=config_p2.downsample,
        window_pre=config_p2.window_pre,
        window_post=config_p2.window_post,
    )

    dVdt_np = compute_dVdt(t_np, V_np,
                           window=config_p2.dVdt_smooth_window,
                           polyorder=config_p2.dVdt_smooth_order)

    # Convert to JAX
    t_ms = jnp.array(t_np)
    V_obs = jnp.array(V_np)
    I_ext_pA = jnp.array(I_np)
    dVdt_obs = jnp.array(dVdt_np)

    print(f"Data: {len(t_ms)} points, {float(t_ms[-1]):.1f}ms", flush=True)

    # ---- Create latent state + conversion factor ----
    latent_gates = LatentGatingState(V_obs)
    conversion = ConversionFactor(membrane_area_cm2=config_p2.membrane_area_cm2_init)

    n_latent = sum(p.size for p in jax.tree.leaves(eqx.filter(latent_gates, eqx.is_array)))
    print(f"Latent gating parameters: {n_latent}", flush=True)
    print(f"Initial membrane area: {float(conversion.membrane_area_um2):.0f} um^2", flush=True)

    # ---- HH reference + sampler for anti-forgetting ----
    hh = HHReference()
    sampler = StateSpaceSampler()

    # ---- Optimizers (separate LRs) ----
    model_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config_p2.model_lr),
    )
    latent_optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(config_p2.latent_lr),
    )
    conv_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config_p2.conversion_lr),
    )

    model_opt_state = model_optimizer.init(eqx.filter(model, eqx.is_array))
    latent_opt_state = latent_optimizer.init(eqx.filter(latent_gates, eqx.is_array))
    conv_opt_state = conv_optimizer.init(eqx.filter(conversion, eqx.is_array))

    # ---- JIT step ----
    step_fn = make_train_step(
        model_optimizer, latent_optimizer, conv_optimizer,
        hh, sampler,
        V_obs, I_ext_pA, dVdt_obs, t_ms,
        field_weight=config_p2.field_weight,
        gating_weight=config_p2.gating_consistency_weight,
        smooth_weight=config_p2.smooth_weight,
        field_batch_size=config_p2.field_batch_size,
    )

    # ---- Training loop ----
    os.makedirs(config_p2.checkpoint_dir, exist_ok=True)
    key = jax.random.PRNGKey(config_p2.seed)
    loss_history = []
    start_time = time.time()
    best_loss = float('inf')

    # ---- Device check ----
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    if jax.default_backend() != 'gpu':
        print("WARNING: Not using GPU! Install jax[cuda12] for CUDA support.")

    # ---- Warmup: trigger JIT compilation ----
    print(f"\nCompiling Phase 2 train step (one-time cost)...", flush=True)
    t_compile = time.time()

    key, warmup_key = jax.random.split(key)
    warmup_result = step_fn(
        model, latent_gates, conversion,
        model_opt_state, latent_opt_state, conv_opt_state,
        warmup_key,
    )
    jax.block_until_ready(warmup_result)
    print(f"Compilation done in {time.time() - t_compile:.1f}s", flush=True)

    print(f"\nTraining: {config_p2.n_epochs} epochs")
    print(f"LRs: model={config_p2.model_lr}, latent={config_p2.latent_lr}, "
          f"conv={config_p2.conversion_lr}")
    print(f"Weights: field={config_p2.field_weight}, "
          f"gating={config_p2.gating_consistency_weight}, "
          f"smooth={config_p2.smooth_weight}")
    print(flush=True)

    for epoch in range(config_p2.n_epochs):
        key, step_key = jax.random.split(key)

        (model, latent_gates, conversion,
         model_opt_state, latent_opt_state, conv_opt_state,
         info) = step_fn(
            model, latent_gates, conversion,
            model_opt_state, latent_opt_state, conv_opt_state,
            step_key,
        )

        # ---- Logging ----
        # Always log epoch 0 to confirm training started
        do_log = (epoch % config_p2.log_every == 0) or epoch == 0
        do_plot = (epoch % config_p2.plot_every == 0) and epoch > 0
        do_ckpt = (epoch % config_p2.checkpoint_every == 0) and epoch > 0

        if do_log or do_plot or do_ckpt:
            info_np = {k: float(v) for k, v in info.items()}
            info_np['epoch'] = epoch
            info_np['membrane_area_um2'] = float(conversion.membrane_area_um2)
            loss_history.append(info_np)

            if do_log:
                elapsed = time.time() - start_time
                dV_mse_str = ""
                if 'dV_mse_raw' in info_np:
                    dV_mse_str = f" | dV_MSE: {info_np['dV_mse_raw']:.1f}"
                print(f"  Epoch {epoch:>5} | "
                      f"Total: {info_np['total_loss']:.6f} | "
                      f"dV: {info_np['dV_loss']:.4f} | "
                      f"Gate: {info_np['gate_loss']:.4f} | "
                      f"Smooth: {info_np['smooth_loss']:.6f} | "
                      f"Field: {info_np['field_loss']:.4f} | "
                      f"Area: {info_np['membrane_area_um2']:.0f}um²"
                      f"{dV_mse_str} | "
                      f"{elapsed:.0f}s", flush=True)

            if do_plot:
                I_ext_hh = conversion.convert(I_ext_pA)
                save_dir = os.path.dirname(config_p2.checkpoint_dir)
                plot_boundary_fit(
                    model, V_obs, I_ext_hh, latent_gates, dVdt_obs, t_ms,
                    epoch=epoch, save_dir=save_dir
                )

            if do_ckpt:
                ckpt_path = os.path.join(
                    config_p2.checkpoint_dir, f"phase2_epoch_{epoch:05d}.eqx"
                )
                eqx.tree_serialise_leaves(ckpt_path, model)

                if info_np['total_loss'] < best_loss:
                    best_loss = info_np['total_loss']
                    eqx.tree_serialise_leaves(
                        os.path.join(config_p2.checkpoint_dir, "phase2_best.eqx"),
                        model
                    )

    # ---- Final ----
    elapsed_total = time.time() - start_time
    print(f"\n--- Phase 2 Complete ({elapsed_total:.0f}s) ---")
    if loss_history:
        print(f"Final total loss: {loss_history[-1]['total_loss']:.6f}")
        print(f"Final dV loss:    {loss_history[-1]['dV_loss']:.6f}")
        print(f"Final gate loss:  {loss_history[-1]['gate_loss']:.6f}")
    print(f"Learned membrane area: {float(conversion.membrane_area_um2):.0f} um² "
          f"(init: {config_p2.membrane_area_cm2_init * 1e8:.0f} um²)")

    # Final plots
    save_dir = os.path.dirname(config_p2.checkpoint_dir)
    I_ext_hh = conversion.convert(I_ext_pA)
    plot_boundary_fit(
        model, V_obs, I_ext_hh, latent_gates, dVdt_obs, t_ms,
        epoch=config_p2.n_epochs, save_dir=save_dir
    )
    plot_phase2_curves(loss_history, save_dir=save_dir)

    # Save
    final_path = os.path.join(config_p2.checkpoint_dir, "phase2_final.eqx")
    eqx.tree_serialise_leaves(final_path, model)
    print(f"Model saved: {final_path}")

    return model, latent_gates, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train Phase 1 first, then Phase 2')
    args = parser.parse_args()

    if args.from_scratch:
        from train_field import train_phase1
        print("Running Phase 1 first...\n")
        model, _ = train_phase1()
        print("\n\n")
        train_phase2(model=model)
    else:
        train_phase2()
