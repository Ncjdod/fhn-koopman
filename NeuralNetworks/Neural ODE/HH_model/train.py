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
import diffrax
import numpy as np
import h5py

from HH_NeuralODE import create_model, integrate
from HodgkinHuxley import HodgkinHuxley
from physics_loss import LossWeights, make_shooting_train_step
from curriculum import CurriculumScheduler
from AllenBrainLoader import download_nwb, find_sweeps, get_sweep_data
from multiple_shooting import compute_segment_boundaries, build_segment_arrays
from visualization import plot_progress, plot_final


class Config:
    """All hyperparameters in one place."""

    # --- Model ---
    n_fourier = 32 
    fourier_sigma = 1.0     
    seed = 42

    # --- Data ---
    downsample_factor = 20
    window_pre = 5.0  
    window_post = 50.0 

    # --- Unit Conversion ---
    # Allen data: pA (picoamperes, absolute current)
    # HH model:   uA/cm2 (current density)
    # Typical cortical soma: ~2000 um^2 = 2e-5 cm^2
    membrane_area_cm2 = 2e-5
    pA_to_uA_per_cm2 = 1e-6 / 2e-5  

    # --- Curriculum ---
    T_start = 5.0 
    T_end = 55.0 
    n_stages = 10
    epochs_per_stage = 300
    schedule = 'linear'
    physics_weight_start = 10.0
    physics_weight_end = 1.0

    # --- Multiple Shooting ---
    n_segments_start = 2 
    n_segments_end = 8 
    n_pts_per_seg = 20
    continuity_weight_start = 0.1
    continuity_weight_end = 10.0 

    # --- Training ---
    model_lr = 1e-3
    weights_lr = 1e-2 
    grad_clip_norm = 1.0 
    log_weight_clamp = 5.0
    n_colloc = 64 
    n_loss_weights = 8 
    log_every = 1 
    plot_every = 500 
    checkpoint_every = 500 
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    val_split = 0.8   

    # --- Integration ---
    dt0 = 0.01
    rtol = 1e-3
    atol = 1e-5

    # --- Adjoint Method ---
    adjoint_method = "recursive_checkpoint"
    adjoint_max_steps = None
    adjoint_rtol = None
    adjoint_atol = None

    @property
    def total_epochs(self):
        return self.n_stages * self.epochs_per_stage


def make_adjoint(config):
    """Create a diffrax adjoint method from configuration."""
    if config.adjoint_method == "backsolve":
        kwargs = {}
        if config.adjoint_max_steps is not None:
            kwargs['max_steps'] = config.adjoint_max_steps
        if config.adjoint_rtol is not None or config.adjoint_atol is not None:
            rtol = config.adjoint_rtol if config.adjoint_rtol is not None else config.rtol
            atol = config.adjoint_atol if config.adjoint_atol is not None else config.atol
            kwargs['stepsize_controller'] = diffrax.PIDController(rtol=rtol, atol=atol)
        return diffrax.BacksolveAdjoint(**kwargs)

    elif config.adjoint_method == "recursive_checkpoint":
        return diffrax.RecursiveCheckpointAdjoint()

    elif config.adjoint_method == "direct":
        return diffrax.DirectAdjoint()

    else:
        raise ValueError(
            f"Unknown adjoint method: {config.adjoint_method}. "
            f"Choose from: 'backsolve', 'recursive_checkpoint', 'direct'"
        )


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
    I_ext_fn_val = make_I_ext_fn(t_val, c_val)
    print(f"Train: {len(t_full)} pts ({float(t_full[-1]):.1f}ms), "
          f"Val: {len(t_val)} pts ({float(t_val[-1] - t_val[0]):.1f}ms)")

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
    weights_optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.weights_lr)
    )

    model_opt_state = model_optimizer.init(eqx.filter(model, eqx.is_array))
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

    # ---- 7. Adjoint Method ----
    adjoint = make_adjoint(config)
    print(f"\nAdjoint method: {config.adjoint_method}")

    # ---- 8. Training Loop ----
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

            train_step_fn = make_shooting_train_step(
                model_optimizer, weights_optimizer,
                hh, all_ics, t_segments, V_segments,
                c_segments, adjoint=adjoint
            )

        if t_segments is None or len(t_sub) < 10:
            continue

        key, ckey1, ckey2 = jax.random.split(key, 3)
        n_colloc = config.n_colloc
        indices = jax.random.randint(ckey1, (n_colloc,), 0, len(t_sub))
        V_colloc = v_sub[indices] + jax.random.normal(ckey2, (n_colloc,)) * 5.0
        t_colloc = t_sub[indices]
        I_colloc_pA = c_sub[indices]
        I_colloc_hh = I_colloc_pA * config.pA_to_uA_per_cm2

        model, loss_weights, model_opt_state, weight_opt_state, info = \
            train_step_fn(
                model, loss_weights,
                model_opt_state, weight_opt_state,
                V_colloc, t_colloc, I_colloc_pA, I_colloc_hh,
                phys_w, cont_w,
            )

        info_np = {k: float(v) for k, v in info.items()}
        info_np['epoch'] = epoch
        info_np['stage'] = stage_num
        info_np['T'] = T_curr
        info_np['n_segments'] = n_seg
        info_np['continuity_weight'] = cont_w

        if epoch % config.log_every == 0 and len(t_val) >= 10:
            y0_val = hh.resting_state(v_val[0])
            y_pred_val = integrate(model, y0_val, t_val, I_ext_fn_val)
            val_loss = float(jnp.mean((y_pred_val[:, 0] - v_val) ** 2))
            info_np['val_loss'] = val_loss

        loss_history.append(info_np)

        if epoch % config.log_every == 0:
            elapsed = time.time() - start_time
            val_str = ""
            if 'val_loss' in info_np:
                val_str = f" | Val: {info_np['val_loss']:>10.4f}"
            print(f"  Epoch {epoch:>5} | "
                  f"Total: {info_np['total_loss']:>10.4f} | "
                  f"Data: {info_np['data_loss']:>10.4f} | "
                  f"Cont: {info_np['continuity_loss']:>10.6f} | "
                  f"Phys: {info_np['physics_loss']:>10.2f} | "
                  f"Segs: {n_seg} | "
                  f"T={T_curr:.1f}ms{val_str} | "
                  f"{elapsed:.0f}s")

        if epoch % config.plot_every == 0 and epoch > 0:
            plot_progress(model, hh, t_full, v_full, c_full, I_ext_fn_full,
                         epoch, info_np, loss_history, n_segments=n_seg)

        if epoch % config.checkpoint_every == 0 and epoch > 0:
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
