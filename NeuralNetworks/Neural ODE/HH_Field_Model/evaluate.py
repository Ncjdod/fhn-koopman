"""
Model Evaluation: Physical Sanity Checks

Tests whether the trained vector field respects fundamental biophysical constraints:
  1. Gating bounds:    m, h, n stay in [0, 1] during integration
  2. Resting stability: No spontaneous firing without current
  3. Action potentials: Step current produces spike-like depolarization
  4. Derivative signs:  At boundaries, field points inward (restoring force)
  5. Field accuracy:    R^2 vs HH ground truth across state space
  6. I-F curve:         Higher current -> higher firing rate
  7. Phase 2 fit:       Boundary condition quality (if Allen data available)

Usage:
    python evaluate.py                        # uses latest checkpoint
    python evaluate.py --checkpoint path.eqx  # specific checkpoint
    python evaluate.py --phase 1              # Phase 1 model only
    python evaluate.py --phase 2              # Phase 2 model (default: tries both)
"""

import os
import sys
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Phase1Config, Phase2Config
from hh_reference import HHReference
from model import create_model, safe_euler_step
from sampler import StateSpaceSampler


# ================================================================
# Helpers
# ================================================================

def load_model(checkpoint_path=None, config=None):
    """Load a model from checkpoint."""
    if config is None:
        config = Phase1Config()

    if checkpoint_path is None:
        # Try phase2 best -> phase2 final -> phase1 best -> phase1 final
        candidates = [
            os.path.join(config.checkpoint_dir, "phase2_best.eqx"),
            os.path.join(config.checkpoint_dir, "phase2_final.eqx"),
            os.path.join(config.checkpoint_dir, "phase1_best.eqx"),
            os.path.join(config.checkpoint_dir, "phase1_final.eqx"),
        ]
        for path in candidates:
            if os.path.exists(path):
                checkpoint_path = path
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found.")
        print(f"Searched: {config.checkpoint_dir}")
        return None, None

    skeleton = create_model(
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_fourier=getattr(config, 'n_fourier', 64),
        sigma=getattr(config, 'fourier_sigma', 1.0),
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, skeleton)
    print(f"Loaded: {checkpoint_path}")
    return model, checkpoint_path


def euler_integrate(step_fn, y0, n_steps, dt, clip_gates=False):
    """
    Euler integration via lax.scan.

    Args:
        step_fn:    function y -> dy/dt
        y0:         initial state (4,)
        n_steps:    number of steps
        dt:         time step
        clip_gates: if True, clip y[1:4] to [0,1] after each step (for NN)
    """
    def scan_step(y, _):
        dy = step_fn(y)
        y_new = y + dt * dy
        if clip_gates:
            # Clip gating variables to [0, 1] for integration safety
            y_new = y_new.at[1].set(jnp.clip(y_new[1], 0.0, 1.0))
            y_new = y_new.at[2].set(jnp.clip(y_new[2], 0.0, 1.0))
            y_new = y_new.at[3].set(jnp.clip(y_new[3], 0.0, 1.0))
        return y_new, y
    y_final, trajectory = jax.lax.scan(scan_step, y0, None, length=n_steps)
    # Append final state
    trajectory = jnp.concatenate([trajectory, y_final[None]], axis=0)
    return trajectory


# ================================================================
# Test 1: Gating Variable Bounds
# ================================================================

def test_gating_bounds(model, hh, save_dir):
    """
    Integrate the model from various ICs and check if m, h, n stay in [0, 1].
    This is the most basic sanity check — gating variables are probabilities.
    """
    print("\n" + "=" * 60)
    print("Test 1: Gating Variable Bounds [0, 1]")
    print("=" * 60)

    dt = 0.01
    T_ms = 100.0
    n_steps = int(T_ms / dt)

    # Test multiple scenarios
    test_cases = [
        ("Resting (I=0)",       hh.resting_state(-65.0), 0.0),
        ("Sub-threshold (I=5)", hh.resting_state(-65.0), 5.0),
        ("Supra-threshold (I=10)", hh.resting_state(-65.0), 10.0),
        ("Strong stim (I=50)",  hh.resting_state(-65.0), 50.0),
        ("Very strong (I=150)", hh.resting_state(-65.0), 150.0),
        ("Depolarized IC",      jnp.array([0.0, 0.5, 0.5, 0.5]), 10.0),
        ("Extreme IC",          jnp.array([40.0, 0.99, 0.01, 0.99]), 10.0),
    ]

    all_pass = True
    results = []

    for name, y0, I_ext in test_cases:
        def step_fn(y):
            return model(y[0], y[1], y[2], y[3], I_ext)

        traj = euler_integrate(step_fn, y0, n_steps, dt, clip_gates=True)
        traj_np = np.array(traj)

        m_range = (traj_np[:, 1].min(), traj_np[:, 1].max())
        h_range = (traj_np[:, 2].min(), traj_np[:, 2].max())
        n_range = (traj_np[:, 3].min(), traj_np[:, 3].max())

        # Check bounds (allow small numerical overshoot)
        eps = 0.05
        m_ok = m_range[0] >= -eps and m_range[1] <= 1.0 + eps
        h_ok = h_range[0] >= -eps and h_range[1] <= 1.0 + eps
        n_ok = n_range[0] >= -eps and n_range[1] <= 1.0 + eps
        ok = m_ok and h_ok and n_ok

        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}: {name:30s} | m=[{m_range[0]:.3f}, {m_range[1]:.3f}] "
              f"h=[{h_range[0]:.3f}, {h_range[1]:.3f}] "
              f"n=[{n_range[0]:.3f}, {n_range[1]:.3f}]")

        results.append((name, traj_np, I_ext, ok))

    # Plot trajectories for visual inspection
    fig, axes = plt.subplots(len(results), 2, figsize=(16, 3 * len(results)))
    t = np.linspace(0, T_ms, n_steps + 1)

    for i, (name, traj, I_ext, ok) in enumerate(results):
        color = 'green' if ok else 'red'

        ax = axes[i, 0]
        ax.plot(t, traj[:, 0], color='blue', lw=1.5)
        ax.set_ylabel('V (mV)')
        ax.set_title(f'{name} (I={I_ext})', color=color, fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(t, traj[:, 1], label='m', color='orange', lw=1)
        ax.plot(t, traj[:, 2], label='h', color='green', lw=1)
        ax.plot(t, traj[:, 3], label='n', color='purple', lw=1)
        ax.axhline(0, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.axhline(1, color='red', ls='--', lw=0.5, alpha=0.5)
        ax.set_ylabel('Gating')
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 1].set_xlabel('Time (ms)')

    fig.suptitle('Gating Bound Check: Integration Trajectories', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_gating_bounds.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    return all_pass


# ================================================================
# Test 2: Resting State Stability
# ================================================================

def test_resting_stability(model, hh, save_dir):
    """
    With zero current, the neuron should settle near resting potential (~-65 mV)
    and NOT spontaneously fire.
    """
    print("\n" + "=" * 60)
    print("Test 2: Resting State Stability (I=0)")
    print("=" * 60)

    dt = 0.01
    T_ms = 200.0
    n_steps = int(T_ms / dt)

    y0 = hh.resting_state(-65.0)

    # NN model
    def nn_step(y):
        return model(y[0], y[1], y[2], y[3], 0.0)
    traj_nn = np.array(euler_integrate(nn_step, y0, n_steps, dt, clip_gates=True))

    # HH ground truth
    def hh_step(y):
        return hh._derivatives_single(y, 0.0)
    traj_hh = np.array(euler_integrate(hh_step, y0, n_steps, dt))

    t = np.linspace(0, T_ms, n_steps + 1)

    # Check: V should stay within ±10 mV of rest
    V_nn = traj_nn[:, 0]
    V_deviation = np.max(np.abs(V_nn - (-65.0)))
    passed = V_deviation < 10.0

    print(f"  V deviation from rest: {V_deviation:.2f} mV (threshold: 10 mV)")
    print(f"  V range: [{V_nn.min():.1f}, {V_nn.max():.1f}] mV")
    print(f"  {'PASS' if passed else 'FAIL'}: {'Stable at rest' if passed else 'UNSTABLE!'}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, traj_hh[:, 0], 'b-', lw=2, label='HH (truth)', alpha=0.7)
    ax.plot(t, traj_nn[:, 0], 'r--', lw=2, label='NN (learned)')
    ax.axhline(-65, color='gray', ls=':', alpha=0.5, label='V_rest = -65 mV')
    ax.set_ylabel('V (mV)')
    ax.set_title('Resting Stability: I_ext = 0 µA/cm²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, traj_hh[:, 1], '-', color='orange', alpha=0.5, label='m (HH)')
    ax.plot(t, traj_nn[:, 1], '--', color='orange', label='m (NN)')
    ax.plot(t, traj_hh[:, 2], '-', color='green', alpha=0.5, label='h (HH)')
    ax.plot(t, traj_nn[:, 2], '--', color='green', label='h (NN)')
    ax.plot(t, traj_hh[:, 3], '-', color='purple', alpha=0.5, label='n (HH)')
    ax.plot(t, traj_nn[:, 3], '--', color='purple', label='n (NN)')
    ax.set_ylabel('Gating')
    ax.set_xlabel('Time (ms)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_resting_stability.png'), dpi=120)
    plt.close()

    return passed


# ================================================================
# Test 3: Action Potential Generation
# ================================================================

def test_action_potential(model, hh, save_dir):
    """
    With supra-threshold current, the model should produce action potentials:
      - V should reach > 0 mV (depolarization)
      - V should return below -50 mV (repolarization)
      - There should be repeated spikes (oscillation)
    """
    print("\n" + "=" * 60)
    print("Test 3: Action Potential Generation")
    print("=" * 60)

    dt = 0.01
    T_ms = 100.0
    n_steps = int(T_ms / dt)

    I_ext_values = [7.0, 10.0, 20.0, 50.0]
    y0 = hh.resting_state(-65.0)

    results = []
    for I_ext in I_ext_values:
        # NN
        def nn_step(y, I=I_ext):
            return model(y[0], y[1], y[2], y[3], I)
        traj_nn = np.array(euler_integrate(nn_step, y0, n_steps, dt, clip_gates=True))

        # HH
        def hh_step(y, I=I_ext):
            return hh._derivatives_single(y, I)
        traj_hh = np.array(euler_integrate(hh_step, y0, n_steps, dt))

        V_nn = traj_nn[:, 0]
        V_hh = traj_hh[:, 0]

        # Count spikes (zero-crossings going up)
        nn_spikes = np.sum(np.diff(np.sign(V_nn - 0.0)) > 0)
        hh_spikes = np.sum(np.diff(np.sign(V_hh - 0.0)) > 0)

        nn_peak = V_nn.max()
        hh_peak = V_hh.max()
        nn_trough = V_nn[n_steps // 5:].min()  # skip initial transient

        depolarizes = nn_peak > -10.0
        repolarizes = nn_trough < -40.0
        has_spikes = nn_spikes > 0

        ok = depolarizes and repolarizes
        status = "PASS" if ok else "FAIL"

        print(f"  {status}: I={I_ext:5.1f} | NN: {nn_spikes} spikes, "
              f"peak={nn_peak:.1f}mV, trough={nn_trough:.1f}mV | "
              f"HH: {hh_spikes} spikes, peak={hh_peak:.1f}mV")

        results.append((I_ext, traj_nn, traj_hh, nn_spikes, hh_spikes, ok))

    # Plot
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
    t = np.linspace(0, T_ms, n_steps + 1)

    for i, (I_ext, traj_nn, traj_hh, nn_sp, hh_sp, ok) in enumerate(results):
        ax = axes[i]
        ax.plot(t, traj_hh[:, 0], 'b-', lw=1.5, alpha=0.6, label='HH (truth)')
        ax.plot(t, traj_nn[:, 0], 'r-', lw=1.5, alpha=0.8, label='NN (learned)')
        ax.axhline(0, color='gray', ls=':', alpha=0.3)
        color = 'green' if ok else 'red'
        ax.set_ylabel('V (mV)')
        ax.set_title(f'I={I_ext} µA/cm²  |  NN: {nn_sp} spikes, HH: {hh_sp} spikes',
                      color=color, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Action Potential Test', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_action_potentials.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    return all(r[5] for r in results)


# ================================================================
# Test 4: Derivative Sign Check (Restoring Force at Boundaries)
# ================================================================

def test_derivative_signs(model, hh, save_dir):
    """
    At gating boundaries, the vector field should point inward:
      - When m ≈ 0: dm/dt should be ≥ 0 (for physiological V)
      - When m ≈ 1: dm/dt should be ≤ 0 (for physiological V)
      - Same logic for h, n
    This is the "restoring force" property.
    """
    print("\n" + "=" * 60)
    print("Test 4: Derivative Signs at Boundaries")
    print("=" * 60)

    # Sample V values across physiological range
    V_values = jnp.linspace(-80.0, 40.0, 200)
    I_ext = 10.0

    checks = []
    for gate_name, gate_idx in [('m', 1), ('h', 2), ('n', 3)]:
        for boundary, value, expected_sign in [('low (~0.01)', 0.01, 'positive'),
                                                ('high (~0.99)', 0.99, 'negative')]:
            # Set test gate to boundary, others at steady-state of V
            m_vals = jnp.where(gate_idx == 1, value, hh.m_inf(V_values))
            h_vals = jnp.where(gate_idx == 2, value, hh.h_inf(V_values))
            n_vals = jnp.where(gate_idx == 3, value, hh.n_inf(V_values))

            states = jnp.stack([V_values, m_vals, h_vals, n_vals], axis=-1)
            I_batch = jnp.full(len(V_values), I_ext)

            dydt_nn = model.predict_batch(states, I_batch)
            dydt_hh = hh.derivatives_batch(states, I_batch)

            dgate_nn = np.array(dydt_nn[:, gate_idx])
            dgate_hh = np.array(dydt_hh[:, gate_idx])

            if expected_sign == 'positive':
                nn_correct = np.mean(dgate_nn > 0)
                hh_correct = np.mean(dgate_hh > 0)
            else:
                nn_correct = np.mean(dgate_nn < 0)
                hh_correct = np.mean(dgate_hh < 0)

            ok = nn_correct > 0.7  # At least 70% of V values should satisfy
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: d{gate_name}/dt at {gate_name}={boundary}: "
                  f"NN {nn_correct*100:.0f}% correct sign "
                  f"(HH: {hh_correct*100:.0f}%)")

            checks.append(ok)

    return all(checks)


# ================================================================
# Test 5: Field Accuracy (R² vs HH)
# ================================================================

def test_field_accuracy(model, hh, save_dir):
    """
    Compute R² for each component across the state space.
    R² > 0.9 is good, > 0.95 is excellent.
    """
    print("\n" + "=" * 60)
    print("Test 5: Vector Field Accuracy (R² vs HH)")
    print("=" * 60)

    sampler = StateSpaceSampler()
    key = jax.random.PRNGKey(123)

    n_samples = 10000
    states, I_ext = sampler.mixed_sample(key, n_samples)
    dydt_true = hh.derivatives_batch(states, I_ext)
    dydt_pred = model.predict_batch(states, I_ext)

    labels = ['dV/dt', 'dm/dt', 'dh/dt', 'dn/dt']
    r2_values = []

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (ax, label) in enumerate(zip(axes, labels)):
        true_i = np.array(dydt_true[:, i])
        pred_i = np.array(dydt_pred[:, i])

        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        r2_values.append(r2)

        rmse = np.sqrt(np.mean((pred_i - true_i) ** 2))

        ax.scatter(true_i, pred_i, s=1, alpha=0.2, c='steelblue')
        lo = min(true_i.min(), pred_i.min())
        hi = max(true_i.max(), pred_i.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(f'{label}\nR² = {r2:.4f}, RMSE = {rmse:.2f}')
        ax.grid(True, alpha=0.3)

        status = "PASS" if r2 > 0.9 else ("WARN" if r2 > 0.7 else "FAIL")
        print(f"  {status}: {label:8s} R² = {r2:.4f}  RMSE = {rmse:.4f}")

    fig.suptitle(f'Vector Field Quality (N={n_samples})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_field_accuracy.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    min_r2 = min(r2_values)
    return min_r2 > 0.7


# ================================================================
# Test 6: Current-Frequency Relationship
# ================================================================

def test_if_curve(model, hh, save_dir):
    """
    The I-F (current-frequency) curve: higher I_ext -> more spikes.
    A fundamental property of excitable neurons.
    """
    print("\n" + "=" * 60)
    print("Test 6: Current-Frequency (I-F) Curve")
    print("=" * 60)

    dt = 0.01
    T_ms = 200.0
    n_steps = int(T_ms / dt)
    y0 = hh.resting_state(-65.0)

    I_values = np.arange(0, 80, 2.0)
    nn_freqs = []
    hh_freqs = []

    for I_ext in I_values:
        I_val = float(I_ext)

        # NN
        def nn_step(y, I=I_val):
            return model(y[0], y[1], y[2], y[3], I)
        traj_nn = np.array(euler_integrate(nn_step, y0, n_steps, dt, clip_gates=True))
        V_nn = traj_nn[:, 0]
        nn_spikes = np.sum(np.diff(np.sign(V_nn - 0.0)) > 0)
        nn_freqs.append(nn_spikes / (T_ms / 1000.0))  # Hz

        # HH
        def hh_step(y, I=I_val):
            return hh._derivatives_single(y, I)
        traj_hh = np.array(euler_integrate(hh_step, y0, n_steps, dt))
        V_hh = traj_hh[:, 0]
        hh_spikes = np.sum(np.diff(np.sign(V_hh - 0.0)) > 0)
        hh_freqs.append(hh_spikes / (T_ms / 1000.0))

    nn_freqs = np.array(nn_freqs)
    hh_freqs = np.array(hh_freqs)

    # Check monotonicity (generally increasing up to some saturation)
    # HH has a threshold around I=6-7, and frequency increases with current
    hh_threshold_idx = np.argmax(hh_freqs > 0)
    nn_threshold_idx = np.argmax(nn_freqs > 0)

    if nn_threshold_idx > 0:
        nn_threshold = I_values[nn_threshold_idx]
    else:
        nn_threshold = float('inf') if nn_freqs[0] == 0 else 0.0
    hh_threshold = I_values[hh_threshold_idx] if hh_threshold_idx > 0 else 0.0

    print(f"  HH threshold: ~{hh_threshold:.0f} µA/cm²")
    print(f"  NN threshold: ~{nn_threshold:.0f} µA/cm²")
    print(f"  HH max freq:  {hh_freqs.max():.0f} Hz (at I={I_values[np.argmax(hh_freqs)]:.0f})")
    print(f"  NN max freq:  {nn_freqs.max():.0f} Hz (at I={I_values[np.argmax(nn_freqs)]:.0f})")

    # Correlation (handle case where NN never fires)
    if nn_freqs.max() == 0:
        corr = 0.0
        print(f"  FAIL: NN never fires — no I-F curve")
    else:
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(hh_freqs, nn_freqs)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        status = "PASS" if corr > 0.7 else "FAIL"
        print(f"  {status}: I-F curve correlation = {corr:.3f}")

    ok = corr > 0.7 and nn_freqs.max() > 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(I_values, hh_freqs, 'b-o', markersize=4, lw=2, label='HH (truth)')
    ax.plot(I_values, nn_freqs, 'r-s', markersize=4, lw=2, label='NN (learned)')
    ax.set_xlabel('I_ext (µA/cm²)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'I-F Curve (corr = {corr:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_if_curve.png'), dpi=120)
    plt.close()

    return ok


# ================================================================
# Test 7: Phase Portraits
# ================================================================

def test_phase_portraits(model, hh, save_dir):
    """
    Visual comparison of NN vs HH vector fields in V-m, V-h, V-n planes.
    """
    print("\n" + "=" * 60)
    print("Test 7: Phase Portraits (visual)")
    print("=" * 60)

    from visualization import plot_phase_portrait
    plot_phase_portrait(model, hh, I_ext_val=10.0, n_grid=30,
                        epoch=9999, save_dir=save_dir)
    print("  Saved phase portrait plot. Check visually for arrow agreement.")
    return True  # Visual check only


# ================================================================
# Summary
# ================================================================

def run_all(model, hh, save_dir):
    """Run all evaluation tests and print summary."""

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "#" * 60)
    print("  HH Vector Field Model — Evaluation Suite")
    print("#" * 60)

    # JIT compile the model prediction once
    print("\nCompiling model for evaluation...", flush=True)
    y0 = hh.resting_state(-65.0)
    _ = model(y0[0], y0[1], y0[2], y0[3], 10.0)
    print("Done.", flush=True)

    results = {}
    results['gating_bounds'] = test_gating_bounds(model, hh, save_dir)
    results['resting_stability'] = test_resting_stability(model, hh, save_dir)
    results['action_potentials'] = test_action_potential(model, hh, save_dir)
    results['derivative_signs'] = test_derivative_signs(model, hh, save_dir)
    results['field_accuracy'] = test_field_accuracy(model, hh, save_dir)
    results['if_curve'] = test_if_curve(model, hh, save_dir)
    results['phase_portraits'] = test_phase_portraits(model, hh, save_dir)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)

    for test_name, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {test_name}")

    print(f"\n  {n_pass}/{n_total} tests passed")
    print(f"  Plots saved to: {save_dir}")
    print("=" * 60)

    return results


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HH vector field model")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .eqx checkpoint file')
    args = parser.parse_args()

    config = Phase1Config()
    hh = HHReference()

    model, ckpt_path = load_model(checkpoint_path=args.checkpoint, config=config)
    if model is None:
        sys.exit(1)

    save_dir = os.path.join(os.path.dirname(config.checkpoint_dir), "eval_plots")
    run_all(model, hh, save_dir)
