"""
HH Neural ODE - Complete Training Script (Multiple Shooting)

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

# Add HH_model dir and parent (Neural ODE) dir to path
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Our modules
from HH_NeuralODE import HHNeuralODE, FourierFeatures, create_model, integrate
from HodgkinHuxley import HodgkinHuxley
from physics_loss import LossWeights, physics_residual, adversarial_physics_loss
from curriculum import CurriculumScheduler
from AllenBrainLoader import download_nwb, find_sweeps, get_sweep_data, SPECIMEN_ID
from multiple_shooting import (
    compute_segment_boundaries, build_segment_arrays,
    integrate_all_segments, shooting_combined_loss,
)


# ============================================================
# Configuration
# ============================================================
class Config:
    """All hyperparameters in one place."""

    # --- Model ---
    n_fourier = 32          # Fourier basis functions
    fourier_sigma = 1.0     # Fourier frequency scale
    seed = 42

    # --- Data ---
    downsample_factor = 20  # 200kHz -> 10kHz
    window_pre = 5.0        # ms before first spike
    window_post = 50.0      # ms after first spike

    # --- Unit Conversion ---
    # Allen data: pA (picoamperes, absolute current)
    # HH model:   uA/cm2 (current density)
    # Typical cortical soma: ~2000 um^2 = 2e-5 cm^2
    membrane_area_cm2 = 2e-5
    pA_to_uA_per_cm2 = 1e-6 / 2e-5  # 0.05 uA/cm2 per pA

    # --- Curriculum ---
    T_start = 5.0           # Start with 5ms window
    T_end = 55.0            # Full window
    n_stages = 10
    epochs_per_stage = 300
    schedule = 'linear'
    physics_weight_start = 10.0
    physics_weight_end = 1.0

    # --- Multiple Shooting ---
    n_segments_start = 2    # Fewer segments for short windows
    n_segments_end = 8      # More segments for long windows
    n_pts_per_seg = 20      # Save points per segment (uniform for vmap)
    continuity_weight_start = 0.1   # Soft start
    continuity_weight_end = 10.0    # Hard finish

    # --- Training ---
    model_lr = 1e-3
    weights_lr = 1e-2       # Weights learn faster (ascent)
    grad_clip_norm = 1.0    # Global gradient norm clipping
    log_weight_clamp = 5.0  # Adversarial log-weight bounds [-5, 5]
    n_colloc = 64           # Collocation points per step
    n_loss_weights = 8      # Adversarial weight bins
    log_every = 1          # Print every N epochs
    plot_every = 500        # Plot every N epochs
    checkpoint_every = 500  # Save checkpoint every N epochs
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    val_split = 0.8         # Fraction of data for training (rest for validation)

    # --- Integration ---
    dt0 = 0.01
    rtol = 1e-3
    atol = 1e-5

    # --- Adjoint Method ---
    # Controls how gradients are backpropagated through the ODE solver.
    # "backsolve":             Continuous adjoint (memory-efficient, approximate gradients)
    #                          NOTE: incompatible with vmap over t_segments (multiple shooting)
    # "recursive_checkpoint":  Discretise-then-optimise (exact gradients, higher memory)
    # "direct":                Standard backprop through solver (no checkpointing)
    adjoint_method = "recursive_checkpoint"
    adjoint_max_steps = None    # Max steps for backward pass (None = same as forward)
    adjoint_rtol = None         # Backward pass rtol (None = same as forward)
    adjoint_atol = None         # Backward pass atol (None = same as forward)

    @property
    def total_epochs(self):
        return self.n_stages * self.epochs_per_stage


# ============================================================
# Adjoint Factory
# ============================================================
def make_adjoint(config):
    """
    Create a diffrax adjoint method from configuration.

    The adjoint method controls how gradients are backpropagated
    through the ODE solver during training.

    Args:
        config: Config instance

    Returns:
        diffrax.AbstractAdjoint instance
    """
    if config.adjoint_method == "backsolve":
        # Continuous adjoint: solves adjoint ODE backwards in time
        # Memory-efficient O(1) but produces approximate gradients
        kwargs = {}
        if config.adjoint_max_steps is not None:
            kwargs['max_steps'] = config.adjoint_max_steps
        if config.adjoint_rtol is not None or config.adjoint_atol is not None:
            rtol = config.adjoint_rtol if config.adjoint_rtol is not None else config.rtol
            atol = config.adjoint_atol if config.adjoint_atol is not None else config.atol
            kwargs['stepsize_controller'] = diffrax.PIDController(rtol=rtol, atol=atol)
        return diffrax.BacksolveAdjoint(**kwargs)

    elif config.adjoint_method == "recursive_checkpoint":
        # Discretise-then-optimise: backprop through solver with checkpointing
        # Exact gradients, O(n log n) memory
        return diffrax.RecursiveCheckpointAdjoint()

    elif config.adjoint_method == "direct":
        # Standard backprop through solver (no checkpointing)
        # Exact gradients, O(n) memory
        return diffrax.DirectAdjoint()

    else:
        raise ValueError(
            f"Unknown adjoint method: {config.adjoint_method}. "
            f"Choose from: 'backsolve', 'recursive_checkpoint', 'direct'"
        )


# ============================================================
# Data Loading
# ============================================================
def load_allen_data(config):
    """Load and preprocess Allen Brain electrophysiology data."""
    print("\n--- Loading Allen Brain Data ---")

    filepath = download_nwb()
    assert filepath is not None, "Failed to download NWB file"

    with h5py.File(filepath, 'r') as f:
        sweeps = find_sweeps(f)
        print(f"Found {len(sweeps)} sweeps")

        # Find best spiking sweep
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

    # Make time relative
    t_raw = t_raw - t_raw[0]

    # Downsample with anti-aliasing (FIR low-pass filter before decimation)
    from scipy.signal import decimate
    ds = config.downsample_factor
    v_ds = decimate(v_raw, ds, ftype='fir')
    c_ds = decimate(c_raw, ds, ftype='fir')
    t_ds = t_raw[::ds][:len(v_ds)]  # Match length after filtering

    # Find first spike and extract window
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
    t_train = t_ds[mask] - t_ds[mask][0]  # Shift to start at 0
    v_train = v_ds[mask]
    c_train = c_ds[mask]

    # Count spikes in window
    cross_win = np.diff(np.sign(v_train - 0.0))
    n_spikes_win = np.sum(cross_win > 0)

    print(f"Training window: {t_train[-1]:.1f}ms, {len(t_train)} points, {n_spikes_win} spikes")
    print(f"V: [{v_train.min():.1f}, {v_train.max():.1f}] mV")
    print(f"I: [{c_train.min():.1f}, {c_train.max():.1f}] pA")

    # Convert to JAX
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


# ============================================================
# Training Step Factory (Minimax with Multiple Shooting)
# ============================================================
def make_shooting_train_step(model_optimizer, weights_optimizer,
                              hh, all_ics, t_segments, V_segments,
                              c_segments, adjoint=None):
    """
    Create a JIT-compiled minimax training step function.

    Uses a factory/closure pattern: static data (HH model, segment arrays,
    optimizers, adjoint method) are captured in the closure rather than
    passed as arguments. This is required for BacksolveAdjoint compatibility
    — eqx.filter_jit traces all arguments as dynamic, but BacksolveAdjoint's
    custom_vjp needs non-array values (like the adjoint and HH objects) to
    remain static constants during tracing.

    The returned function only accepts values that change every step:
    model params, loss weights, optimizer states, and collocation points.

    Must be rebuilt when segment arrays change (at curriculum stage boundaries).

    Args:
        model_optimizer:    Optax optimizer for model (gradient descent)
        weights_optimizer:  Optax optimizer for loss weights (gradient ascent)
        hh:                 HodgkinHuxley instance (fixed)
        all_ics:            (K, 4) data-pinned initial conditions
        t_segments:         (K, n_pts_per_seg) time arrays per segment
        V_segments:         (K, n_pts_per_seg) target voltage per segment
        c_segments:         (K, n_pts_per_seg) current arrays in pA per segment
        adjoint:            Diffrax adjoint method (None = RecursiveCheckpointAdjoint)

    Returns:
        step_fn:  JIT-compiled function with signature:
                  (model, loss_weights, model_opt_state, weights_opt_state,
                   V_colloc, t_colloc, I_colloc_model, I_colloc_hh,
                   physics_weight, continuity_weight)
                  -> (model, loss_weights, model_opt_state, weights_opt_state, info)
    """
    @eqx.filter_jit
    def step(model, loss_weights,
             model_opt_state, weights_opt_state,
             V_colloc, t_colloc, I_colloc_model, I_colloc_hh,
             physics_weight, continuity_weight):
        """
        Single minimax training step with multiple shooting.

        All K segments integrate in parallel via jax.vmap.
        ICs are data-pinned (from Allen data, not trainable).
        Gradients through ODE solver use the configured adjoint method.

        Two gradient computations:
          1. Model params:  gradient DESCENT (minimize total loss)
          2. Loss weights:  gradient ASCENT  (maximize physics loss)
        """
        # --- Model gradients (descent) ---
        @eqx.filter_value_and_grad(has_aux=True)
        def model_loss(model):
            return shooting_combined_loss(
                model, loss_weights, hh,
                all_ics, t_segments, V_segments, c_segments,
                V_colloc, t_colloc, I_colloc_model, I_colloc_hh,
                physics_weight, continuity_weight,
                adjoint=adjoint
            )

        (loss, info), model_grads = model_loss(model)

        # --- Weight gradients (ascent) ---
        @eqx.filter_value_and_grad(has_aux=True)
        def weight_loss(loss_weights):
            return shooting_combined_loss(
                model, loss_weights, hh,
                all_ics, t_segments, V_segments, c_segments,
                V_colloc, t_colloc, I_colloc_model, I_colloc_hh,
                physics_weight, continuity_weight,
                adjoint=adjoint
            )

        (_, _), weight_grads = weight_loss(loss_weights)

        model_updates, model_opt_state_new = model_optimizer.update(
            model_grads, model_opt_state, model
        )
        model = eqx.apply_updates(model, model_updates)

        neg_weight_grads = jax.tree.map(lambda g: -g, weight_grads)
        weight_updates, weights_opt_state_new = weights_optimizer.update(
            neg_weight_grads, weights_opt_state, loss_weights
        )
        loss_weights = eqx.apply_updates(loss_weights, weight_updates)

        loss_weights = eqx.tree_at(
            lambda lw: lw.log_weights,
            loss_weights,
            jnp.clip(loss_weights.log_weights, -5.0, 5.0)
        )

        return model, loss_weights, model_opt_state_new, weights_opt_state_new, info

    return step


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


# ============================================================
# Main Training Loop
# ============================================================
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

    # Count parameters
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
        alpha=0.01  # Final LR = 1e-5
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

    # Pre-declare segment arrays and step function (built once per stage)
    t_segments = V_segments = c_segments = all_ics = None
    t_sub = v_sub = c_sub = None
    train_step_fn = None  # JIT-compiled step, rebuilt each stage

    for epoch in range(config.total_epochs):
        # Get curriculum parameters
        stage = scheduler.get_stage(epoch)
        T_curr = stage['T']
        phys_w = stage['physics_weight']
        cont_w = stage['continuity_weight']
        n_seg = stage['n_segments']
        n_pts = stage['n_pts_per_seg']
        stage_num = stage['stage']

        # On stage change: rebuild segment arrays AND step function
        # (shapes change here → new JIT compilation, once per stage)
        if stage_num != prev_stage:
            print(f"\n>> Stage {stage_num}: T={T_curr:.1f}ms, "
                  f"phys_w={phys_w:.2f}, cont_w={cont_w:.2f}, "
                  f"segments={n_seg}")
            prev_stage = stage_num

            # Get curriculum sub-window and build segments ONCE per stage
            t_sub, v_sub, c_sub = get_curriculum_data(t_full, v_full, c_full, T_curr)
            if len(t_sub) < 10:
                continue
            boundaries = compute_segment_boundaries(t_sub, n_seg)
            t_segments, V_segments, c_segments, all_ics = build_segment_arrays(
                t_sub, v_sub, c_sub, boundaries, n_pts, hh
            )

            # Rebuild JIT step function with new segment arrays in closure.
            # Static data (hh, optimizers, adjoint, segment arrays, t_full/c_full)
            # are captured in the closure — required for BacksolveAdjoint
            # compatibility (avoids DynamicJaxprTracer errors from eqx.filter_jit
            # tracing non-array objects like HodgkinHuxley and BacksolveAdjoint).
            train_step_fn = make_shooting_train_step(
                model_optimizer, weights_optimizer,
                hh, all_ics, t_segments, V_segments,
                c_segments, adjoint=adjoint
            )

        # Skip if data too sparse (from stage init above)
        if t_segments is None or len(t_sub) < 10:
            continue

        # Sample collocation points from trajectory (re-randomized each epoch)
        key, ckey1, ckey2 = jax.random.split(key, 3)
        n_colloc = config.n_colloc
        indices = jax.random.randint(ckey1, (n_colloc,), 0, len(t_sub))
        V_colloc = v_sub[indices] + jax.random.normal(ckey2, (n_colloc,)) * 5.0
        t_colloc = t_sub[indices]
        I_colloc_pA = c_sub[indices]
        I_colloc_hh = I_colloc_pA * config.pA_to_uA_per_cm2

        # Minimax step with multiple shooting (continuous adjoint for gradients)
        # Collocation points and curriculum weights are the only per-epoch args.
        # All other data (segments, t_full/c_full, hh, adjoint) is in the closure.
        model, loss_weights, model_opt_state, weight_opt_state, info = \
            train_step_fn(
                model, loss_weights,
                model_opt_state, weight_opt_state,
                V_colloc, t_colloc, I_colloc_pA, I_colloc_hh,
                phys_w, cont_w,
            )

        # Log
        info_np = {k: float(v) for k, v in info.items()}
        info_np['epoch'] = epoch
        info_np['stage'] = stage_num
        info_np['T'] = T_curr
        info_np['n_segments'] = n_seg
        info_np['continuity_weight'] = cont_w

        # Compute validation loss periodically
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

        # Plot progress
        if epoch % config.plot_every == 0 and epoch > 0:
            plot_progress(model, hh, t_full, v_full, c_full, I_ext_fn_full,
                         epoch, info_np, loss_history, n_segments=n_seg)

        # Checkpoint + track best model
        if epoch % config.checkpoint_every == 0 and epoch > 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch:05d}.eqx")
            eqx.tree_serialise_leaves(ckpt_path, model)

        if info_np['data_loss'] < best_data_loss:
            best_data_loss = info_np['data_loss']
            eqx.tree_serialise_leaves(
                os.path.join(config.checkpoint_dir, "best_model.eqx"), model
            )

    # ---- 9. Final Results ----
    elapsed_total = time.time() - start_time
    print(f"\n--- Training Complete ({elapsed_total:.0f}s) ---")
    print(f"Final data loss:       {loss_history[-1]['data_loss']:.6f}")
    print(f"Final continuity loss: {loss_history[-1]['continuity_loss']:.6f}")
    print(f"Final physics loss:    {loss_history[-1]['physics_loss']:.4f}")

    # Final plot
    plot_final(model, hh, t_full, v_full, c_full, I_ext_fn_full, loss_history)

    # Save final model (absolute path based on script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "trained_model.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    print(f"Final model saved to {model_path}")

    # Report best model location
    best_model_path = os.path.join(
        os.path.abspath(config.checkpoint_dir), "best_model.eqx"
    )
    print(f"Best model (data loss {best_data_loss:.6f}) at {best_model_path}")

    return model, loss_history


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    model, history = train()
