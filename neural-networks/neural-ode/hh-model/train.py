"""
HH Neural ODE - Training Script (Multiple Shooting)

Combines all components:
  - Allen Brain data loading & preprocessing
  - HH Neural ODE model (4D state: V, m, h, n)
  - Adversarial physics loss (trainable weights, gradient ascent)
  - Curriculum learning (progressive time windows + segment ramp)
  - Multiple shooting (parallel segment integration via jax.vmap)
  - Minimax training loop with gradient clipping + LR scheduling

Usage:
    python train.py
"""

import os
import sys
import time

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, '..'))
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import h5py

from HH_NeuralODE import create_model
from HodgkinHuxley import HodgkinHuxley
from physics_loss import LossWeights, PhysicsParams, make_shooting_train_step
from curriculum import CurriculumScheduler
from AllenBrainLoader import download_nwb, find_sweeps, get_sweep_data
from multiple_shooting import compute_segment_boundaries, build_segment_arrays, _heun_scan_integrate
from visualization import plot_progress, plot_final


class Config:
    """All hyperparameters in one place."""

    # --- Model ---
    n_fourier = 32
    fourier_sigma = 10.0
    seed = 42

    # --- Data ---
    downsample_factor = 20
    window_pre = 5.0  
    window_post = 50.0 

    # --- Unit Conversion (trainable) ---
    # Allen data: pA (picoamperes, absolute current)
    # HH model:   uA/cm2 (current density)
    # Initial estimate: ~2000 um^2 = 2e-5 cm^2 (typical cortical soma)
    # The actual membrane area is learned via physics loss gradient
    membrane_area_cm2_init = 2e-5

    # --- Curriculum ---
    T_start = 5.0 
    T_end = 55.0 
    n_stages = 10
    epochs_per_stage = 300
    schedule = 'linear'
    physics_weight_start = 0.0
    physics_weight_end = 1.0

    # --- Multiple Shooting ---
    n_segments_start = 10
    n_segments_end = 30
    n_pts_per_seg = 5
    continuity_weight_start = 0.1
    continuity_weight_end = 10.0 

    # --- Training ---
    model_lr = 1e-3
    physics_lr = 1e-3
    weights_lr = 1e-2
    grad_clip_norm = 1.0 
    log_weight_clamp = 3.0
    n_colloc = 64 
    n_loss_weights = 8 
    log_every = 10
    val_every = 50
    plot_every = 500
    checkpoint_every = 500
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    val_split = 0.8   

    # --- Integration ---
    dt0 = 0.01
    rtol = 1e-3
    atol = 1e-5

    # (Integration is handled by custom Heun scan integrator in multiple_shooting.py)

    @property
    def total_epochs(self):
        return self.n_stages * self.epochs_per_stage


def load_allen_data(config):
    """Load and preprocess Allen Brain electrophysiology data."""
    print("\n--- Loading Allen Brain Data ---")

    filepath = download_nwb()
    assert filepath is not None, "Failed to download NWB file"

    with h5py.File(filepath, 'r') as f:
        sweeps = find_sweeps(f)
        print(f"Found {len(sweeps)} sweeps")

        best_sweep = None
        best_n_spikes = 0

        for sweep_name in sweeps:
            t, v, c = get_sweep_data(f, sweep_name)
            if v is None:
                continue
            crossings = np.diff(np.sign(v - 0.0))
            n_spikes = np.sum(crossings > 0)
            if n_spikes > best_n_spikes:
                best_n_spikes = n_spikes
                best_sweep = sweep_name

        print(f"Best sweep: {best_sweep} ({best_n_spikes} spikes)")
        t_raw, v_raw, c_raw = get_sweep_data(f, best_sweep)

    t_raw = t_raw - t_raw[0]

    from scipy.signal import decimate
    ds = config.downsample_factor
    v_ds = decimate(v_raw, ds, ftype='fir')
    c_ds = decimate(c_raw, ds, ftype='fir')
    t_ds = t_raw[::ds][:len(v_ds)] 

    crossings = np.diff(np.sign(v_ds - 0.0))
    spike_idx = np.where(crossings > 0)[0]

    if len(spike_idx) > 0:
        first_spike_t = t_ds[spike_idx[0]]
        t_start = max(first_spike_t - config.window_pre, t_ds[0])
        t_end = min(first_spike_t + config.window_post, t_ds[-1])
    else:
        print("WARNING: No spikes found. Using first 55ms.")
        t_start = t_ds[0]
        t_end = t_ds[0] + 55.0

    mask = (t_ds >= t_start) & (t_ds <= t_end)
    t_train = t_ds[mask] - t_ds[mask][0]
    v_train = v_ds[mask]
    c_train = c_ds[mask]

    cross_win = np.diff(np.sign(v_train - 0.0))
    n_spikes_win = np.sum(cross_win > 0)

    print(f"Training window: {t_train[-1]:.1f}ms, {len(t_train)} points, {n_spikes_win} spikes")
    print(f"V: [{v_train.min():.1f}, {v_train.max():.1f}] mV")
    print(f"I: [{c_train.min():.1f}, {c_train.max():.1f}] pA")

    t_jax = jnp.array(t_train, dtype=jnp.float32)
    v_jax = jnp.array(v_train, dtype=jnp.float32)
    c_jax = jnp.array(c_train, dtype=jnp.float32)

    return t_jax, v_jax, c_jax


def make_I_ext_fn(t_data, c_data):
    """Create interpolation function for external current."""
    def I_ext_fn(t):
        return jnp.interp(t, t_data, c_data)
    return I_ext_fn


def get_curriculum_data(t_full, v_full, c_full, T_window):
    """
    Extract a sub-window [0, T_window] from the full training data.
    Used by curriculum to progressively expand the window.
    """
    mask = t_full <= T_window
    t_sub = t_full[mask]
    v_sub = v_full[mask]
    c_sub = c_full[mask]
    return t_sub, v_sub, c_sub


def train(config=None):
    """Run the full training pipeline with multiple shooting."""
    if config is None:
        config = Config()

    print("=" * 60)
    print("HH Neural ODE - Training (Multiple Shooting)")
    print("=" * 60)

    key = jax.random.PRNGKey(config.seed)

    # ---- 1. Load Data ----
    t_all, v_all, c_all = load_allen_data(config)

    # Train/validation split (temporal holdout)
    n_train = int(config.val_split * len(t_all))
    t_full, v_full, c_full = t_all[:n_train], v_all[:n_train], c_all[:n_train]
    t_val, v_val, c_val = t_all[n_train:], v_all[n_train:], c_all[n_train:]
    I_ext_fn_full = make_I_ext_fn(t_full, c_full)
    print(f"Train: {len(t_full)} pts ({float(t_full[-1]):.1f}ms), "
          f"Val: {len(t_val)} pts ({float(t_val[-1] - t_val[0]):.1f}ms)")

    # JIT-compiled validation function using custom Heun integrator
    @eqx.filter_jit
    def val_fn(model, y0):
        return _heun_scan_integrate(model, y0, t_val, c_val, n_substeps=1)

    # ---- 2. Create Model ----
    print("\n--- Creating Model ---")
    key, model_key = jax.random.split(key)
    model = create_model(
        key=model_key,
        n_fourier=config.n_fourier,
        sigma=config.fourier_sigma
    )

    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model parameters: {n_params}")

    # ---- 3. Create Adversarial Loss Weights ----
    loss_weights = LossWeights(
        n_terms=config.n_colloc,
        init_value=0.0
    )
    print(f"Adversarial weights: {config.n_colloc} terms")

    # ---- 3b. Trainable Physics Parameters ----
    physics_params = PhysicsParams(
        membrane_area_cm2=config.membrane_area_cm2_init
    )
    print(f"Trainable membrane area: {float(physics_params.membrane_area_um2):.0f} um2 (initial)")

    # ---- 4. Physics Model ----
    hh = HodgkinHuxley()

    # ---- 5. Optimizers (gradient clipping + LR scheduling) ----
    model_lr_schedule = optax.cosine_decay_schedule(
        init_value=config.model_lr,
        decay_steps=config.total_epochs,
        alpha=0.01
    )
    model_optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(model_lr_schedule)
    )
    physics_optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.physics_lr)
    )
    weights_optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.weights_lr)
    )

    model_opt_state = model_optimizer.init(eqx.filter(model, eqx.is_array))
    physics_opt_state = physics_optimizer.init(
        eqx.filter(physics_params, eqx.is_array)
    )
    weight_opt_state = weights_optimizer.init(
        eqx.filter(loss_weights, eqx.is_array)
    )

    # ---- 6. Curriculum Scheduler ----
    scheduler = CurriculumScheduler(
        T_start=config.T_start,
        T_end=min(config.T_end, float(t_full[-1])),
        n_stages=config.n_stages,
        epochs_per_stage=config.epochs_per_stage,
        schedule=config.schedule,
        physics_weight_start=config.physics_weight_start,
        physics_weight_end=config.physics_weight_end,
        n_segments_start=config.n_segments_start,
        n_segments_end=config.n_segments_end,
        continuity_weight_start=config.continuity_weight_start,
        continuity_weight_end=config.continuity_weight_end,
        n_pts_per_seg=config.n_pts_per_seg,
    )

    print("\n--- Curriculum Schedule ---")
    scheduler.summary()

    # ---- 7. Training Loop ----
    print(f"\n--- Training ({config.total_epochs} epochs, multiple shooting) ---")
    loss_history = []
    start_time = time.time()
    prev_stage = -1
    best_data_loss = float('inf')
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    t_segments = V_segments = c_segments = all_ics = None
    t_sub = v_sub = c_sub = None
    train_step_fn = None

    for epoch in range(config.total_epochs):
        stage = scheduler.get_stage(epoch)
        T_curr = stage['T']
        phys_w = stage['physics_weight']
        cont_w = stage['continuity_weight']
        n_seg = stage['n_segments']
        n_pts = stage['n_pts_per_seg']
        stage_num = stage['stage']

        if stage_num != prev_stage:
            print(f"\n>> Stage {stage_num}: T={T_curr:.1f}ms, "
                  f"phys_w={phys_w:.2f}, cont_w={cont_w:.2f}, "
                  f"segments={n_seg}")
            prev_stage = stage_num

            t_sub, v_sub, c_sub = get_curriculum_data(t_full, v_full, c_full, T_curr)
            if len(t_sub) < 10:
                continue
            boundaries = compute_segment_boundaries(t_sub, n_seg)
            t_segments, V_segments, c_segments, all_ics = build_segment_arrays(
                t_sub, v_sub, c_sub, boundaries, n_pts, hh
            )

            # Fixed collocation indices for this stage (static grid)
            key, colloc_key = jax.random.split(key)
            n_total_pts = n_seg * n_pts
            colloc_indices = jax.random.choice(
                colloc_key, n_total_pts,
                shape=(config.n_colloc,), replace=True
            )

            train_step_fn = make_shooting_train_step(
                model_optimizer, physics_optimizer, weights_optimizer,
                hh, all_ics, t_segments, V_segments,
                c_segments, colloc_indices,
            )

        if t_segments is None or len(t_sub) < 10:
            continue

        # Convert to JAX arrays so eqx.filter_jit treats them as dynamic
        # (Python floats are treated as static → recompilation every epoch)
        (model, physics_params, loss_weights,
         model_opt_state, physics_opt_state, weight_opt_state, info) = \
            train_step_fn(
                model, physics_params, loss_weights,
                model_opt_state, physics_opt_state, weight_opt_state,
                jnp.array(phys_w), jnp.array(cont_w),
            )

        # Only materialize values when we need to log (avoids GPU sync every epoch)
        do_log = (epoch % config.log_every == 0)
        do_val = (epoch % config.val_every == 0) and len(t_val) >= 10
        do_checkpoint = (epoch % config.checkpoint_every == 0) and epoch > 0
        do_plot = (epoch % config.plot_every == 0) and epoch > 0

        if do_log or do_val or do_checkpoint:
            info_np = {k: float(v) for k, v in info.items()}
            info_np['epoch'] = epoch
            info_np['stage'] = stage_num
            info_np['T'] = T_curr
            info_np['n_segments'] = n_seg
            info_np['continuity_weight'] = cont_w

            if do_val:
                y0_val = hh.resting_state(v_val[0])
                y_pred_val = val_fn(model, y0_val)
                val_loss = float(jnp.mean((y_pred_val[:, 0] - v_val) ** 2))
                info_np['val_loss'] = val_loss

            loss_history.append(info_np)

            if do_log:
                elapsed = time.time() - start_time
                val_str = ""
                if 'val_loss' in info_np:
                    val_str = f" | Val: {info_np['val_loss']:>10.4f}"
                area_str = ""
                if 'membrane_area_um2' in info_np:
                    area_str = f" | Area: {info_np['membrane_area_um2']:>7.0f}um2"
                print(f"  Epoch {epoch:>5} | "
                      f"Total: {info_np['total_loss']:>10.4f} | "
                      f"Data: {info_np['data_loss']:>10.4f} | "
                      f"Cont: {info_np['continuity_loss']:>10.6f} | "
                      f"Phys: {info_np['physics_loss']:>10.2f} | "
                      f"Segs: {n_seg} | "
                      f"T={T_curr:.1f}ms{val_str}{area_str} | "
                      f"{elapsed:.0f}s")

            if do_plot:
                plot_progress(model, hh, t_full, v_full, c_full, I_ext_fn_full,
                             epoch, info_np, loss_history, n_segments=n_seg)

            if do_checkpoint:
                ckpt_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch:05d}.eqx")
                eqx.tree_serialise_leaves(ckpt_path, model)
                if info_np['data_loss'] < best_data_loss:
                    best_data_loss = info_np['data_loss']
                    eqx.tree_serialise_leaves(
                        os.path.join(config.checkpoint_dir, "best_model.eqx"), model
                    )

    elapsed_total = time.time() - start_time
    print(f"\n--- Training Complete ({elapsed_total:.0f}s) ---")
    print(f"Final data loss:       {loss_history[-1]['data_loss']:.6f}")
    print(f"Final continuity loss: {loss_history[-1]['continuity_loss']:.6f}")
    print(f"Final physics loss:    {loss_history[-1]['physics_loss']:.4f}")
    print(f"Learned membrane area: {float(physics_params.membrane_area_um2):.0f} um2 "
          f"(init: {config.membrane_area_cm2_init * 1e8:.0f} um2)")

    plot_final(model, hh, t_full, v_full, c_full, I_ext_fn_full, loss_history)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "trained_model.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    print(f"Final model saved to {model_path}")

    best_model_path = os.path.join(
        os.path.abspath(config.checkpoint_dir), "best_model.eqx"
    )
    print(f"Best model (data loss {best_data_loss:.6f}) at {best_model_path}")

    return model, loss_history


if __name__ == "__main__":
    model, history = train()
