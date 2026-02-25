"""
Multiple Shooting for Neural ODE Training

Splits the time domain into K segments, each starting from known Allen data.
All segments integrate independently via jax.vmap for GPU parallelism.
A continuity loss penalizes mismatches at segment boundaries.

Key design:
  - ICs are data-pinned (not trainable) — enables full parallelism
  - Fixed n_pts_per_seg for uniform array shapes (vmap requirement)
  - Continuity loss: ||y_end_integrated[k] - y_data_start[k+1]||^2
  - 4D state: [V, m, h, n] — gating variables from HH steady-state at data V
  - Dual current units: model sees pA, HH physics sees uA/cm2
"""

import jax
import jax.numpy as jnp
import diffrax

from HH_NeuralODE import integrate
from HodgkinHuxley import HodgkinHuxley
from physics_loss import adversarial_physics_loss


def compute_segment_boundaries(t_data, n_segments):
    """
    Compute evenly spaced boundary times dividing [t_data[0], t_data[-1]].

    Args:
        t_data:     Full time array
        n_segments: Number of segments K

    Returns:
        boundaries: (K+1,) array of boundary times
    """
    return jnp.linspace(t_data[0], t_data[-1], n_segments + 1)


def build_segment_arrays(t_data, V_data, c_data, boundaries, n_pts_per_seg, hh):
    """
    Build uniform per-segment arrays for vmap integration.

    Each segment gets n_pts_per_seg evenly spaced time points.
    Target voltage and current are interpolated from the full data.
    Initial conditions are 4D: [V, m_inf(V), h_inf(V), n_inf(V)] from data.

    Args:
        t_data:        Full time array
        V_data:        Full voltage data
        c_data:        Full current data (pA)
        boundaries:    (K+1,) segment boundary times
        n_pts_per_seg: Number of save points per segment
        hh:            HodgkinHuxley instance (for steady-state gating)

    Returns:
        t_segments: (K, n_pts_per_seg) — time arrays
        V_segments: (K, n_pts_per_seg) — target voltage
        c_segments: (K, n_pts_per_seg) — current data (pA)
        all_ics:    (K, 4) — initial conditions [V, m_inf, h_inf, n_inf]
    """
    n_segments = len(boundaries) - 1

    t_segs = []
    V_segs = []
    c_segs = []
    ics = []

    for k in range(n_segments):
        t_seg = jnp.linspace(boundaries[k], boundaries[k + 1], n_pts_per_seg)
        V_seg = jnp.interp(t_seg, t_data, V_data)
        c_seg = jnp.interp(t_seg, t_data, c_data)

        t_segs.append(t_seg)
        V_segs.append(V_seg)
        c_segs.append(c_seg)

        V0 = V_seg[0]
        ics.append(hh.resting_state(V0))

    t_segments = jnp.stack(t_segs)   # (K, n_pts_per_seg)
    V_segments = jnp.stack(V_segs)   # (K, n_pts_per_seg)
    c_segments = jnp.stack(c_segs)   # (K, n_pts_per_seg)
    all_ics = jnp.stack(ics)         # (K, 4)

    return t_segments, V_segments, c_segments, all_ics



def integrate_all_segments(model, all_ics, t_segments, c_segments,
                           dt0=0.01, rtol=1e-3, atol=1e-5, max_steps=4096,
                           adjoint=None):
    """
    Integrate all K segments in parallel using jax.vmap.

    Each segment uses the shared model but independent IC and time span.
    External current is interpolated from the segment data arrays.

    Args:
        model:       HHNeuralODE (shared across segments)
        all_ics:     (K, 4) initial conditions [V, m, h, n] from data
        t_segments:  (K, n_pts_per_seg) time arrays
        c_segments:  (K, n_pts_per_seg) current arrays in pA
        dt0:         Initial step size
        rtol, atol:  Tolerances
        max_steps:   Max ODE solver steps per segment
        adjoint:     Diffrax adjoint method (None = RecursiveCheckpointAdjoint).
                     Static PyTree, safe to capture in vmap closure.

    Returns:
        all_trajectories: (K, n_pts_per_seg, 4) predicted trajectories
    """
    def _single_segment(y0, t_span, c_span):
        I_ext_fn = lambda t: jnp.interp(t, t_span, c_span)
        return integrate(model, y0, t_span, I_ext_fn,
                         dt0=dt0, rtol=rtol, atol=atol, max_steps=max_steps,
                         adjoint=adjoint)

    return jax.vmap(_single_segment)(all_ics, t_segments, c_segments)


def continuity_loss(all_trajectories, all_ics):
    """
    Penalize discontinuities at segment boundaries.

    Measures: mean ||y_end_integrated[k] - y_data_start[k+1]||^2
    for k = 0, ..., K-2.  Compares all 4 state variables.

    Args:
        all_trajectories: (K, n_pts_per_seg, 4)
        all_ics:          (K, 4) — data-pinned initial conditions

    Returns:
        loss: scalar
    """
    y_ends = all_trajectories[:-1, -1, :]     # (K-1, 4)
    y_next_starts = all_ics[1:, :]            # (K-1, 4)

    gaps = y_ends - y_next_starts
    return jnp.mean(gaps ** 2)


def shooting_data_loss(all_trajectories, V_segments):
    """
    MSE data loss on voltage across all segments.

    Args:
        all_trajectories: (K, n_pts_per_seg, 4)
        V_segments:       (K, n_pts_per_seg) — target voltage only

    Returns:
        loss: scalar MSE
    """
    V_pred = all_trajectories[:, :, 0]   # (K, n_pts_per_seg)
    return jnp.mean((V_pred - V_segments) ** 2)


def shooting_gating_penalty(all_trajectories):
    """
    Penalize gating variables outside [0, 1].

    Args:
        all_trajectories: (K, n_pts_per_seg, 4)

    Returns:
        penalty: scalar
    """
    m = all_trajectories[:, :, 1]
    h = all_trajectories[:, :, 2]
    n = all_trajectories[:, :, 3]
    penalty = jnp.max(
        jnp.maximum(0.0, -m) + jnp.maximum(0.0, m - 1.0) +
        jnp.maximum(0.0, -h) + jnp.maximum(0.0, h - 1.0) +
        jnp.maximum(0.0, -n) + jnp.maximum(0.0, n - 1.0)
    )
    return penalty


def shooting_combined_loss(model, loss_weights, hh,
                           all_ics, t_segments, V_segments, c_segments,
                           V_colloc, t_colloc, I_colloc_model, I_colloc_hh,
                           physics_weight, continuity_weight,
                           adjoint=None):
    """
    Full multiple-shooting loss:
        L = data_loss + continuity_weight * continuity_loss
                      + physics_weight * adversarial_physics_loss
                      + 10.0 * gating_penalty

    Args:
        model:             HHNeuralODE (gradient descent)
        loss_weights:      LossWeights (gradient ascent)
        hh:                HodgkinHuxley (fixed)
        all_ics:           (K, 4) data-pinned initial conditions
        t_segments:        (K, n_pts_per_seg) time arrays
        V_segments:        (K, n_pts_per_seg) target voltage
        c_segments:        (K, n_pts_per_seg) current arrays in pA
        V_colloc, t_colloc: Collocation points for physics loss
        I_colloc_model:    Collocation current in pA (for neural ODE)
        I_colloc_hh:       Collocation current in uA/cm2 (for HH)
        physics_weight:    Scalar weight for physics loss
        continuity_weight: Scalar weight for continuity loss
        adjoint:           Diffrax adjoint method (None = RecursiveCheckpointAdjoint)

    Returns:
        total_loss: scalar
        info: dict with component losses
    """
    all_trajs = integrate_all_segments(model, all_ics, t_segments, c_segments,
                                       adjoint=adjoint)

    d_loss = shooting_data_loss(all_trajs, V_segments)
    c_loss = continuity_loss(all_trajs, all_ics)
    g_penalty = shooting_gating_penalty(all_trajs)

    p_loss, p_info = adversarial_physics_loss(
        model, loss_weights, hh, V_colloc, t_colloc, I_colloc_model, I_colloc_hh
    )

    total = (d_loss
             + continuity_weight * c_loss
             + physics_weight * p_loss
             + 10.0 * g_penalty)

    info = {
        'total_loss': total,
        'data_loss': d_loss,
        'continuity_loss': c_loss,
        'gating_penalty': g_penalty,
        'physics_loss': p_info['physics_loss'],
        'weighted_phys': p_info['weighted_loss'],
        'mean_weight': p_info['mean_weight'],
        'max_weight': p_info['max_weight'],
    }

    return total, info


if __name__ == "__main__":
    from HH_NeuralODE import create_model
    from physics_loss import LossWeights

    print("Multiple Shooting - Test (4D State)")
    print("=" * 50)

    key = jax.random.PRNGKey(0)
    model = create_model(key=key)
    hh = HodgkinHuxley()

    # Fake data: 10ms, 100 points
    t_data = jnp.linspace(0.0, 10.0, 100)
    V_data = -65.0 + 5.0 * jnp.sin(2 * jnp.pi * t_data / 10.0)
    c_data = jnp.ones(100) * 200.0 

    n_segments = 3
    n_pts_per_seg = 20

    boundaries = compute_segment_boundaries(t_data, n_segments)
    print(f"Boundaries: {boundaries}")

    t_segs, V_segs, c_segs, all_ics = build_segment_arrays(
        t_data, V_data, c_data, boundaries, n_pts_per_seg, hh
    )
    print(f"t_segments shape: {t_segs.shape}")
    print(f"V_segments shape: {V_segs.shape}")
    print(f"all_ics shape:    {all_ics.shape}")  # (3, 4)

    print("\nRunning parallel integration...")
    all_trajs = integrate_all_segments(model, all_ics, t_segs, c_segs)
    print(f"all_trajectories shape: {all_trajs.shape}")  # (3, 20, 4)

    c_loss = continuity_loss(all_trajs, all_ics)
    print(f"Continuity loss: {c_loss:.6f}")

    d_loss = shooting_data_loss(all_trajs, V_segs)
    print(f"Data loss: {d_loss:.6f}")

    loss_weights = LossWeights(n_terms=64, init_value=0.0)
    V_colloc = jax.random.uniform(key, (64,), minval=-80.0, maxval=40.0)
    t_colloc = jax.random.uniform(key, (64,), minval=0.0, maxval=10.0)
    I_colloc_pA = jnp.ones(64) * 200.0
    pA_to_uA_per_cm2 = 1e-6 / 2e-5
    I_colloc_hh = I_colloc_pA * pA_to_uA_per_cm2

    total, info = shooting_combined_loss(
        model, loss_weights, hh,
        all_ics, t_segs, V_segs, c_segs,
        V_colloc, t_colloc, I_colloc_pA, I_colloc_hh,
        physics_weight=1.0, continuity_weight=1.0
    )
    print(f"\nCombined loss: {total:.6f}")
    for k, v in info.items():
        print(f"  {k}: {float(v):.6f}")

    print("\nMultiple Shooting OK!")
