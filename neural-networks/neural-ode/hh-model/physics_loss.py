"""
Adversarial Physics Loss & All Loss/Optimization Functions

Implements:
  - Trainable loss weights (Self-Adaptive PINN, gradient ASCENT)
  - Physics residual comparing Neural ODE vs HH derivatives
  - Multiple shooting loss components (data, continuity, gating)
  - Combined shooting loss
  - Factory for JIT-compiled minimax training step

Minimax formulation:
  L(theta, s) = sum_i exp(s_i) * R_i(theta) - sum_i s_i

  Model params theta: gradient DESCENT (minimize L)
  Loss weights s:     gradient ASCENT  (maximize L)

Reference: McClenny & Brainerd, "Self-Adaptive PINNs" (2020)
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from HodgkinHuxley import HodgkinHuxley
from multiple_shooting import integrate_all_segments


class PhysicsParams(eqx.Module):
    """
    Trainable physics parameters learned jointly from data + physics loss.

    Currently holds the membrane area (as log-conversion factor), which
    controls the pA -> uA/cm2 unit mapping between Allen data and HH equations.
    Optimized via gradient DESCENT alongside the model.

    The conversion factor only appears in the physics loss (HH current scaling),
    so its gradient comes from the physics loss — it's learned from the physics
    structure, not just data fitting.
    """
    log_conversion: jnp.ndarray  # scalar: log(pA_to_uA_per_cm2)

    def __init__(self, membrane_area_cm2=2e-5):
        """
        Args:
            membrane_area_cm2: Initial membrane surface area estimate (cm2).
                               Default 2e-5 cm2 = 2000 um2 (typical cortical soma).
        """
        conversion = 1e-6 / membrane_area_cm2  # pA -> uA/cm2
        self.log_conversion = jnp.array(jnp.log(conversion))

    @property
    def pA_to_uA_per_cm2(self):
        """Current conversion factor (always positive via exp)."""
        return jnp.exp(self.log_conversion)

    @property
    def membrane_area_cm2(self):
        """Implied membrane area in cm2."""
        return 1e-6 / self.pA_to_uA_per_cm2

    @property
    def membrane_area_um2(self):
        """Implied membrane area in um2 (more intuitive scale)."""
        return self.membrane_area_cm2 * 1e8


class LossWeights(eqx.Module):
    """
    Trainable loss weights for adversarial physics training.

    Stores log-weights s_i. Actual weights are exp(s_i),
    ensuring positivity. Updated via gradient ASCENT.
    """
    log_weights: jnp.ndarray  # (n_terms,)

    def __init__(self, n_terms, init_value=0.0):
        """
        Args:
            n_terms:    Number of loss terms to weight
            init_value: Initial log-weight (0.0 means weight=1.0)
        """
        self.log_weights = jnp.ones(n_terms) * init_value

    @property
    def weights(self):
        """Actual (positive) weights."""
        return jnp.exp(self.log_weights)

    def regularization(self):
        """
        Regularization term: -sum(s_i)
        Prevents weights from collapsing to zero.
        """
        return -jnp.sum(self.log_weights)


def physics_residual(model, hh, y_samples, t_samples, I_ext_model, I_ext_hh):
    """
    Compute physics residual at collocation points.

    Compares Neural ODE derivatives against HH derivatives at the given
    states. States should come from the simulated trajectory so that
    gating variables reflect actual dynamics, not steady-state.
    Feature-wise scaling balances voltage and gating contributions.

    Args:
        model:       HHNeuralODE instance
        hh:          HodgkinHuxley instance
        y_samples:   Full state [V, m, h, n] at collocation points (N, 4)
        t_samples:   Time at collocation points (N,)
        I_ext_model: Current for neural ODE in pA (N,)
        I_ext_hh:    Current for HH in uA/cm2 (N,)

    Returns:
        residuals: Per-point mean scaled squared error (N,)
    """
    # Neural ODE predictions (pA current units)
    pred_derivatives = jax.vmap(model)(t_samples, y_samples, I_ext_model)

    # HH ground truth (uA/cm2 current units)
    true_derivatives = jax.vmap(
        lambda t, y, I: hh(t, y, I)
    )(t_samples, y_samples, I_ext_hh)

    raw_squared_errors = jnp.square(pred_derivatives - true_derivatives)

    # Scale gating derivatives up to balance against voltage
    scale_weights = jnp.array([1.0, 100.0, 100.0, 100.0])
    scaled_errors = raw_squared_errors * scale_weights

    residuals = jnp.mean(scaled_errors, axis=-1)

    return residuals


def adversarial_physics_loss(model, loss_weights, hh,
                              y_samples, t_samples, I_ext_model, I_ext_hh):
    """
    Self-Adaptive physics loss with trainable weights.

    L = sum_i exp(s_i) * R_i - sum_i s_i

    The -s_i regularization prevents weights from going to infinity.

    Args:
        model:       HHNeuralODE instance
        loss_weights: LossWeights instance (trainable)
        hh:          HodgkinHuxley instance
        y_samples:   Full state [V, m, h, n] at collocation points (N, 4)
        t_samples:   Time collocation points (N,)
        I_ext_model: External current for neural ODE, in pA (N,)
        I_ext_hh:    External current for HH equations, in uA/cm2 (N,)

    Returns:
        total_loss: Scalar loss (to be minimized by model, maximized by weights)
        info:       Dict with diagnostic information
    """
    residuals = physics_residual(model, hh, y_samples, t_samples, I_ext_model, I_ext_hh)

    weights = loss_weights.weights

    if weights.shape[0] < residuals.shape[0]:
        n_per_weight = residuals.shape[0] // weights.shape[0]
        weights = jnp.repeat(weights, n_per_weight)[:residuals.shape[0]]

    weighted_residuals = weights * residuals

    loss = jnp.mean(weighted_residuals) + loss_weights.regularization()

    info = {
        'physics_loss': jnp.mean(residuals),
        'weighted_loss': jnp.mean(weighted_residuals),
        'mean_weight': jnp.mean(weights),
        'max_weight': jnp.max(weights),
        'min_weight': jnp.min(weights),
        '_residuals': residuals,
    }

    return loss, info



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


def shooting_gating_penalty(all_trajectories):
    """
    Penalize gating variables outside [0, 1].

    Uses mean penalty for smooth gradients across all violations,
    rather than max which only backpropagates through a single point.

    Args:
        all_trajectories: (K, n_pts_per_seg, 4)

    Returns:
        penalty: scalar
    """
    m = all_trajectories[:, :, 1]
    h = all_trajectories[:, :, 2]
    n = all_trajectories[:, :, 3]
    violations = (
        jnp.maximum(0.0, -m) + jnp.maximum(0.0, m - 1.0) +
        jnp.maximum(0.0, -h) + jnp.maximum(0.0, h - 1.0) +
        jnp.maximum(0.0, -n) + jnp.maximum(0.0, n - 1.0)
    )
    return jnp.mean(violations)


def shooting_combined_loss(model, physics_params, loss_weights, hh,
                           all_ics, t_segments, V_segments, c_segments,
                           colloc_indices,
                           physics_weight, continuity_weight):
    """
    Full multiple-shooting loss:
        L = data_loss + continuity_weight * continuity_loss
                      + physics_weight * adversarial_physics_loss
                      + 10.0 * gating_penalty

    Collocation points for physics loss are sampled from the simulated
    trajectory at fixed indices (static per curriculum stage), so gating
    variables reflect actual dynamics rather than steady-state values.

    The pA -> uA/cm2 conversion is handled by physics_params (trainable).

    Args:
        model:             HHNeuralODE (gradient descent)
        physics_params:    PhysicsParams (gradient descent, trainable conversion)
        loss_weights:      LossWeights (gradient ascent)
        hh:                HodgkinHuxley (fixed)
        all_ics:           (K, 4) data-pinned initial conditions
        t_segments:        (K, n_pts_per_seg) time arrays
        V_segments:        (K, n_pts_per_seg) target voltage
        c_segments:        (K, n_pts_per_seg) current arrays in pA
        colloc_indices:    (n_colloc,) fixed indices into flattened trajectory
        physics_weight:    Scalar weight for physics loss
        continuity_weight: Scalar weight for continuity loss
        adjoint:           Diffrax adjoint method (None = RecursiveCheckpointAdjoint)

    Returns:
        total_loss: scalar
        info: dict with component losses
    """
    all_trajs = integrate_all_segments(model, all_ics, t_segments, c_segments)

    d_loss = shooting_data_loss(all_trajs, V_segments)
    c_loss = continuity_loss(all_trajs, all_ics)
    g_penalty = shooting_gating_penalty(all_trajs)

    # Sample collocation states from the simulated trajectory.
    # Stop-gradient on trajectory states: physics loss gradient flows only
    # through the direct model(t,y,I) call, NOT through the integration chain.
    # This prevents gradient explosion from chaining Jacobians through
    # all Heun steps (the data loss still backprops through integration).
    t_flat = t_segments.reshape(-1)           # (K*P,)
    y_flat = jax.lax.stop_gradient(all_trajs.reshape(-1, 4))  # (K*P, 4)
    c_flat = c_segments.reshape(-1)           # (K*P,)

    t_colloc = t_flat[colloc_indices]
    y_colloc = y_flat[colloc_indices]         # (n_colloc, 4) - real gating!
    I_colloc_pA = c_flat[colloc_indices]
    I_colloc_hh = I_colloc_pA * physics_params.pA_to_uA_per_cm2

    p_loss, p_info = adversarial_physics_loss(
        model, loss_weights, hh, y_colloc, t_colloc, I_colloc_pA, I_colloc_hh
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
        '_residuals': p_info['_residuals'],
    }

    return total, info



def make_shooting_train_step(model_optimizer, physics_optimizer,
                              weights_optimizer,
                              hh, all_ics, t_segments, V_segments,
                              c_segments, colloc_indices):
    """
    Create a JIT-compiled minimax training step function.

    Single forward+backward pass for model/physics_params gradients.
    Weight gradients are computed analytically from stopped-gradient
    residuals (no second integration needed).

    Integration uses a custom Heun integrator via jax.lax.scan —
    no diffrax overhead, exact step count, zero backward-pass padding.

    Args:
        model_optimizer:    Optax optimizer for model (gradient descent)
        physics_optimizer:  Optax optimizer for physics params (gradient descent)
        weights_optimizer:  Optax optimizer for loss weights (gradient ascent)
        hh:                 HodgkinHuxley instance (fixed)
        all_ics:            (K, 4) data-pinned initial conditions
        t_segments:         (K, n_pts_per_seg) time arrays per segment
        V_segments:         (K, n_pts_per_seg) target voltage per segment
        c_segments:         (K, n_pts_per_seg) current arrays in pA per segment
        colloc_indices:     (n_colloc,) fixed indices into flattened trajectory

    Returns:
        step_fn:  JIT-compiled function
    """
    @eqx.filter_jit
    def step(model, physics_params, loss_weights,
             model_opt_state, physics_opt_state, weights_opt_state,
             physics_weight, continuity_weight):

        # --- Single forward+backward pass for model + physics_params ---
        @eqx.filter_value_and_grad(has_aux=True)
        def model_physics_loss(args):
            model, physics_params = args
            return shooting_combined_loss(
                model, physics_params, loss_weights, hh,
                all_ics, t_segments, V_segments, c_segments,
                colloc_indices,
                physics_weight, continuity_weight,
            )

        (loss, info), (model_grads, physics_grads) = \
            model_physics_loss((model, physics_params))

        # --- Weight gradients from stopped-gradient residuals ---
        # ∂L/∂s_i = exp(s_i) * R_i / N - 1   (no integration needed)
        residuals_sg = jax.lax.stop_gradient(info['_residuals'])
        weights = loss_weights.weights
        if weights.shape[0] < residuals_sg.shape[0]:
            n_per_weight = residuals_sg.shape[0] // weights.shape[0]
            weights = jnp.repeat(weights, n_per_weight)[:residuals_sg.shape[0]]
        weight_grads_raw = physics_weight * (weights * residuals_sg) / residuals_sg.shape[0] - 1.0
        # Negate for ascent (optimizer does descent, we want ascent)
        neg_weight_grads = LossWeights(n_terms=loss_weights.log_weights.shape[0])
        neg_weight_grads = eqx.tree_at(
            lambda lw: lw.log_weights,
            neg_weight_grads,
            -weight_grads_raw[:loss_weights.log_weights.shape[0]]
        )

        # --- Non-finite gradient cleaning ---
        # Handles NaN AND inf: inf in grads causes clip_by_global_norm
        # to produce 0 * inf = NaN, corrupting model weights.
        def clean_grad(x):
            return jnp.where(jnp.isfinite(x), x, 0.0)
        model_grads = jax.tree.map(clean_grad, model_grads)
        physics_grads = jax.tree.map(clean_grad, physics_grads)
        neg_weight_grads = jax.tree.map(clean_grad, neg_weight_grads)

        # --- Apply updates ---
        model_updates, model_opt_state_new = model_optimizer.update(
            model_grads, model_opt_state, model
        )
        model = eqx.apply_updates(model, model_updates)

        physics_updates, physics_opt_state_new = physics_optimizer.update(
            physics_grads, physics_opt_state, physics_params
        )
        physics_params = eqx.apply_updates(physics_params, physics_updates)
        physics_params = eqx.tree_at(
            lambda pp: pp.log_conversion,
            physics_params,
            jnp.clip(physics_params.log_conversion, -4.6, 2.3)
        )

        weight_updates, weights_opt_state_new = weights_optimizer.update(
            neg_weight_grads, weights_opt_state, loss_weights
        )
        loss_weights = eqx.apply_updates(loss_weights, weight_updates)
        loss_weights = eqx.tree_at(
            lambda lw: lw.log_weights,
            loss_weights,
            jnp.clip(loss_weights.log_weights, -3.0, 3.0)
        )

        info['membrane_area_um2'] = physics_params.membrane_area_um2
        info['pA_to_uA'] = physics_params.pA_to_uA_per_cm2
        # Strip internal array from returned info
        info.pop('_residuals', None)

        return (model, physics_params, loss_weights,
                model_opt_state_new, physics_opt_state_new,
                weights_opt_state_new, info)

    return step


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from HH_NeuralODE import create_model
    import optax

    print("Adversarial Physics Loss - Test")
    print("=" * 50)

    key = jax.random.PRNGKey(0)

    # Create components
    model = create_model(key=key)
    hh = HodgkinHuxley()

    # Create trainable loss weights
    n_weights = 8
    loss_weights = LossWeights(n_terms=n_weights, init_value=0.0)
    print(f"Initial weights: {loss_weights.weights}")

    # Sample collocation points with full state from trajectory
    N = 64
    key1, key2, key3, key4 = jax.random.split(key, 4)
    V_colloc = jax.random.uniform(key1, (N,), minval=-80.0, maxval=40.0)
    t_colloc = jax.random.uniform(key2, (N,), minval=0.0, maxval=100.0)
    # Build full 4D state (simulating what trajectory sampling does)
    m_vals = jax.vmap(HodgkinHuxley.m_inf)(V_colloc)
    h_vals = jax.vmap(HodgkinHuxley.h_inf)(V_colloc)
    n_vals = jax.vmap(HodgkinHuxley.n_inf)(V_colloc)
    y_colloc = jnp.stack([V_colloc, m_vals, h_vals, n_vals], axis=-1)  # (N, 4)
    I_colloc_pA = jnp.ones(N) * 200.0  # 200 pA (typical Allen stimulus)
    pA_to_uA_per_cm2 = 1e-6 / 2e-5     # ~2000 um^2 soma
    I_colloc_hh = I_colloc_pA * pA_to_uA_per_cm2  # ~10 uA/cm2

    # Compute loss
    loss, info = adversarial_physics_loss(
        model, loss_weights, hh, y_colloc, t_colloc, I_colloc_pA, I_colloc_hh
    )
    print(f"\nInitial loss: {loss:.4f}")
    print(f"Physics residual: {info['physics_loss']:.4f}")
    print(f"Weight range: [{info['min_weight']:.3f}, {info['max_weight']:.3f}]")

    print("\nAdversarial Physics Loss OK!")
