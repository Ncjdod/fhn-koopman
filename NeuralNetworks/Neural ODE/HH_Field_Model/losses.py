"""
Loss Functions for HH Vector Field Learning

Phase 1 — Field Loss:
    Variance-normalized MSE between NN-predicted and HH-true derivatives.
    Auto-balances V derivatives (~100s mV/ms) vs gating derivatives (~1-10 /ms).

Phase 2 — Boundary Loss:
    Consistency along the Allen Brain trajectory:
    1. dV/dt consistency: predicted dV/dt matches observed (finite-diff) dV/dt
    2. Gating consistency: predicted dm/dt matches finite-diff of latent m(t)
    3. Smoothness: penalize jerky latent gating trajectories
    + Anti-forgetting field loss on fresh HH samples
"""

import jax
import jax.numpy as jnp
import equinox as eqx


# ================================================================
# Phase 1: Vector Field Distillation Loss
# ================================================================

def field_loss(model, states, I_ext, dydt_true):
    """
    Variance-normalized MSE between predicted and true derivatives.

    L = mean_over_samples( sum_over_components( ((dy_pred - dy_true) / sigma)^2 ) )

    where sigma = std(dy_true) per component, computed on the current batch.
    This auto-balances the loss across components with very different scales
    (dV/dt ~ O(100) mV/ms vs dm/dt ~ O(1) /ms).

    Args:
        model:     VectorFieldNet
        states:    (N, 4) — [V, m, h, n] per sample
        I_ext:     (N,) — external current per sample
        dydt_true: (N, 4) — ground truth from HH equations

    Returns:
        loss:     scalar (mean normalized MSE)
        info:     dict with diagnostics
    """
    dydt_pred = model.predict_batch(states, I_ext)  # (N, 4)

    # Per-component variance for normalization
    sigma = jnp.std(dydt_true, axis=0)               # (4,)
    sigma = jnp.maximum(sigma, 1e-6)                  # avoid division by zero

    # Normalized residuals
    residuals = (dydt_pred - dydt_true) / sigma       # (N, 4)
    per_component = jnp.mean(residuals ** 2, axis=0)  # (4,)
    loss = jnp.mean(per_component)

    # Raw (unnormalized) MSE for monitoring
    raw_mse = jnp.mean((dydt_pred - dydt_true) ** 2, axis=0)  # (4,)

    info = {
        "field_loss": loss,
        "mse_dV": raw_mse[0],
        "mse_dm": raw_mse[1],
        "mse_dh": raw_mse[2],
        "mse_dn": raw_mse[3],
        "sigma_dV": sigma[0],
        "sigma_dm": sigma[1],
        "sigma_dh": sigma[2],
        "sigma_dn": sigma[3],
    }

    return loss, info


# ================================================================
# Phase 2: Boundary Condition Loss
# ================================================================

def _finite_diff(x, dt):
    """
    Central finite differences for interior points, forward/backward at edges.

    Args:
        x:  (T,) array
        dt: (T-1,) array of time steps, or scalar

    Returns:
        dx_dt: (T,) array of derivatives
    """
    if jnp.ndim(dt) == 0:
        dt = jnp.full(len(x) - 1, dt)

    # Central differences for interior
    dx_central = (x[2:] - x[:-2]) / (dt[:-1] + dt[1:])

    # Forward difference at start
    dx_fwd = (x[1] - x[0]) / dt[0]

    # Backward difference at end
    dx_bwd = (x[-1] - x[-2]) / dt[-1]

    return jnp.concatenate([dx_fwd[None], dx_central, dx_bwd[None]])


def boundary_loss(model, V_obs, I_ext_hh, latent_gates, dVdt_obs, t_ms):
    """
    Consistency loss along the observed Allen Brain trajectory.

    Evaluates the learned vector field at each trajectory point
    (V_obs(t), m_latent(t), h_latent(t), n_latent(t), I_ext(t))
    and checks consistency with the observed dynamics.

    Components:
        1. dV/dt match:    ||f_V(state) - dV/dt_observed||^2
        2. Gating match:   ||f_gate(state) - d(gate_latent)/dt||^2
        3. Smoothness:     ||d^2(gate_latent)/dt^2||^2

    Args:
        model:        VectorFieldNet
        V_obs:        (T,) observed voltage in mV
        I_ext_hh:     (T,) external current in uA/cm^2 (already converted)
        latent_gates: LatentGatingState module (.m, .h, .n are (T,) arrays)
        dVdt_obs:     (T,) observed dV/dt (from Savitzky-Golay)
        t_ms:         (T,) time points in ms

    Returns:
        loss:  scalar
        info:  dict with component losses
    """
    T = len(V_obs)
    dt = jnp.diff(t_ms)  # (T-1,)

    # Build full state at each time point
    m_lat = latent_gates.m  # (T,)
    h_lat = latent_gates.h
    n_lat = latent_gates.n

    states = jnp.stack([V_obs, m_lat, h_lat, n_lat], axis=-1)  # (T, 4)

    # Predict vector field at all trajectory points
    dydt_pred = model.predict_batch(states, I_ext_hh)  # (T, 4)

    # --- 1. dV/dt consistency (variance-normalized) ---
    dVdt_pred = dydt_pred[:, 0]
    dV_sigma = jnp.maximum(jnp.std(dVdt_obs), 1e-6)
    dV_loss = jnp.mean(((dVdt_pred - dVdt_obs) / dV_sigma) ** 2)

    # --- 2. Gating consistency (variance-normalized) ---
    # Finite-diff derivatives of the latent gating variables
    dm_obs = _finite_diff(m_lat, dt)
    dh_obs = _finite_diff(h_lat, dt)
    dn_obs = _finite_diff(n_lat, dt)

    dm_pred = dydt_pred[:, 1]
    dh_pred = dydt_pred[:, 2]
    dn_pred = dydt_pred[:, 3]

    dm_sigma = jnp.maximum(jnp.std(dm_obs), 1e-6)
    dh_sigma = jnp.maximum(jnp.std(dh_obs), 1e-6)
    dn_sigma = jnp.maximum(jnp.std(dn_obs), 1e-6)

    gate_loss = (jnp.mean(((dm_pred - dm_obs) / dm_sigma) ** 2) +
                 jnp.mean(((dh_pred - dh_obs) / dh_sigma) ** 2) +
                 jnp.mean(((dn_pred - dn_obs) / dn_sigma) ** 2)) / 3.0

    # --- 3. Smoothness of latent variables ---
    # Penalize second derivative (jerkiness), normalized by scale
    d2m = jnp.diff(m_lat, n=2)
    d2h = jnp.diff(h_lat, n=2)
    d2n = jnp.diff(n_lat, n=2)

    smooth_loss = (jnp.mean(d2m ** 2) +
                   jnp.mean(d2h ** 2) +
                   jnp.mean(d2n ** 2)) / 3.0

    # Raw (unnormalized) MSE for monitoring
    dV_mse_raw = jnp.mean((dVdt_pred - dVdt_obs) ** 2)

    info = {
        "dV_loss": dV_loss,
        "dV_mse_raw": dV_mse_raw,
        "gate_loss": gate_loss,
        "smooth_loss": smooth_loss,
    }

    return dV_loss, gate_loss, smooth_loss, info


def total_phase2_loss(model, latent_gates, conversion_factor,
                      V_obs, I_ext_pA, dVdt_obs, t_ms,
                      hh, sampler, sampler_key,
                      field_weight=0.1,
                      gating_weight=1.0,
                      smooth_weight=0.01,
                      field_batch_size=2048):
    """
    Full Phase 2 loss: boundary condition + anti-forgetting field loss.

    L = dV_loss
      + gating_weight * gate_loss
      + smooth_weight * smooth_loss
      + field_weight  * field_loss (on fresh random HH samples)

    The field_loss term prevents catastrophic forgetting of the general
    vector field while the model adapts to the boundary data.

    Args:
        model:            VectorFieldNet
        latent_gates:     LatentGatingState
        conversion_factor: ConversionFactor module
        V_obs:            (T,) voltage in mV
        I_ext_pA:         (T,) current in pA (raw Allen units)
        dVdt_obs:         (T,) observed dV/dt in mV/ms
        t_ms:             (T,) time in ms
        hh:               HHReference instance
        sampler:          StateSpaceSampler instance
        sampler_key:      JAX PRNG key for sampling
        field_weight:     Weight for anti-forgetting field loss
        gating_weight:    Weight for gating consistency
        smooth_weight:    Weight for latent smoothness
        field_batch_size: Batch size for field loss sampling

    Returns:
        total_loss: scalar
        info:       dict with all component losses
    """
    # Convert Allen current to HH units
    I_ext_hh = conversion_factor.convert(I_ext_pA)  # pA -> uA/cm^2

    # Boundary loss components
    dV_loss, gate_loss, smooth_loss, boundary_info = boundary_loss(
        model, V_obs, I_ext_hh, latent_gates, dVdt_obs, t_ms
    )

    # Anti-forgetting field loss on fresh HH samples
    states_fresh, I_fresh = sampler.mixed_sample(sampler_key, field_batch_size)
    dydt_true = hh.derivatives_batch(states_fresh, I_fresh)
    fld_loss, field_info = field_loss(model, states_fresh, I_fresh, dydt_true)

    # Total
    total = (dV_loss
             + gating_weight * gate_loss
             + smooth_weight * smooth_loss
             + field_weight * fld_loss)

    info = {
        "total_loss": total,
        "dV_loss": dV_loss,
        "gate_loss": gate_loss,
        "smooth_loss": smooth_loss,
        "field_loss": fld_loss,
        "gating_weight": gating_weight,
        "smooth_weight": smooth_weight,
        "field_weight": field_weight,
    }

    return total, info
