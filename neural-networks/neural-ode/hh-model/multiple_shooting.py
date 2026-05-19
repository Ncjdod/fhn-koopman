"""
Multiple Shooting - Segment Building & Parallel Integration

Splits the time domain into K segments, each starting from known Allen data.
All segments integrate independently via jax.vmap for GPU parallelism.

Key design:
  - ICs are data-pinned (not trainable) — enables full parallelism
  - Fixed n_pts_per_seg for uniform array shapes (vmap requirement)
  - 4D state: [V, m, h, n] — gating variables from HH steady-state at data V
  - Dual current units: model sees pA, HH physics sees uA/cm2
  - Custom Heun integrator via jax.lax.scan — no diffrax overhead,
    exact step count, zero max_steps padding in backward pass

Loss functions and optimization are in physics_loss.py.
"""

import jax
import jax.numpy as jnp


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


def _heun_scan_integrate(model, y0, t_save, c_save, n_substeps):
    """
    Heun (2nd-order RK) integrator via jax.lax.scan.

    No diffrax overhead. Exact step count (n_save_pts - 1) * n_substeps.
    Zero padding in backward pass.

    Takes n_substeps Heun steps between each consecutive pair of save
    times. Current is linearly interpolated from the save-time arrays.

    Args:
        model:      HHNeuralODE — called as model(t, y, I_ext)
        y0:         (4,) initial state
        t_save:     (P,) save times (evenly spaced)
        c_save:     (P,) current at save times (pA)
        n_substeps: Number of Heun steps between each pair of save times

    Returns:
        ys: (P, 4) trajectory at save times (includes y0 at t_save[0])
    """
    n_intervals = t_save.shape[0] - 1

    def interval_step(y, interval_data):
        """Integrate from t_save[i] to t_save[i+1] with n_substeps Heun steps."""
        t_start, t_end, c_start, c_end = interval_data
        dt = (t_end - t_start) / n_substeps
        sub_ts = jnp.linspace(t_start, t_end, n_substeps + 1)
        sub_cs = jnp.linspace(c_start, c_end, n_substeps + 1)

        def heun_step(y, i):
            t_i = sub_ts[i]
            c_i = sub_cs[i]
            t_next = sub_ts[i + 1]
            c_next = sub_cs[i + 1]

            k1 = model(t_i, y, c_i)
            y_euler = y + dt * k1
            k2 = model(t_next, y_euler, c_next)
            y_new = y + 0.5 * dt * (k1 + k2)
            # Safety clamp: prevent NaN propagation from extreme derivatives
            V = jnp.clip(y_new[0], -200.0, 200.0)
            gates = jnp.clip(y_new[1:], -0.5, 1.5)
            y_new = jnp.concatenate([V[None], gates])
            return y_new, None

        y_end, _ = jax.lax.scan(heun_step, y, jnp.arange(n_substeps))
        return y_end, y_end

    interval_data = (
        t_save[:-1],     # t_start: (n_intervals,)
        t_save[1:],      # t_end:   (n_intervals,)
        c_save[:-1],     # c_start: (n_intervals,)
        c_save[1:],      # c_end:   (n_intervals,)
    )

    _, ys_after = jax.lax.scan(interval_step, y0, interval_data)
    # Prepend initial state
    ys = jnp.concatenate([y0[None, :], ys_after], axis=0)  # (P, 4)
    return ys


def integrate_all_segments(model, all_ics, t_segments, c_segments,
                           n_substeps=1, **_ignored):
    """
    Integrate all K segments in parallel using jax.vmap.

    Uses a custom Heun integrator with jax.lax.scan. Each save-time
    interval gets n_substeps Heun steps, giving a total of
    (n_pts_per_seg - 1) * n_substeps steps per segment. Step count is
    exact — no max_steps padding, no diffrax overhead.

    Args:
        model:       HHNeuralODE (shared across segments)
        all_ics:     (K, 4) initial conditions
        t_segments:  (K, n_pts_per_seg) time arrays
        c_segments:  (K, n_pts_per_seg) current arrays (pA)
        n_substeps:  Heun steps between each pair of save times (default 4)

    Returns:
        all_trajectories: (K, n_pts_per_seg, 4) predicted trajectories
    """
    def _single_segment(y0, t_span, c_span):
        return _heun_scan_integrate(model, y0, t_span, c_span, n_substeps)

    return jax.vmap(_single_segment)(all_ics, t_segments, c_segments)


if __name__ == "__main__":
    from HH_NeuralODE import create_model
    from HodgkinHuxley import HodgkinHuxley

    print("Multiple Shooting - Test (Segment Building & Integration)")
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

    print("\nMultiple Shooting (Segment Utils) OK!")
