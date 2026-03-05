"""
Hodgkin-Huxley Reference Model (Batch-Vectorized)

Provides the ground-truth vector field for Phase 1 training.
All functions work on scalars or batches via jnp broadcasting.
A vmap'd batch interface is provided for efficient training data generation.

Units: mV (voltage), ms (time), uA/cm^2 (current density), mS/cm^2 (conductance)
"""

import jax
import jax.numpy as jnp


class HHReference:
    """
    Standard Hodgkin-Huxley neuron model, optimized for batch evaluation.

    Usage:
        hh = HHReference()

        # Single point:
        dV, dm, dh, dn = hh.derivatives(V, m, h, n, I_ext)

        # Batch (N points):
        states = jnp.array([[V, m, h, n], ...])  # (N, 4)
        I_ext  = jnp.array([...])                 # (N,)
        dydt   = hh.derivatives_batch(states, I_ext)  # (N, 4)
    """

    # ---- Membrane parameters ----
    C_m = 1.0       # uF/cm^2

    # ---- Maximum conductances (mS/cm^2) ----
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3

    # ---- Reversal potentials (mV) ----
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.4

    # ================================================================
    # Rate functions (alpha, beta)
    # ================================================================

    @staticmethod
    def alpha_m(V):
        dV = V + 40.0
        safe_dV = jnp.where(jnp.abs(dV) < 1e-6, 1.0, dV)
        return jnp.where(
            jnp.abs(dV) < 1e-6,
            1.0,
            0.1 * safe_dV / (1.0 - jnp.exp(-safe_dV / 10.0))
        )

    @staticmethod
    def beta_m(V):
        return 4.0 * jnp.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V):
        return 0.07 * jnp.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def beta_h(V):
        return 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def alpha_n(V):
        dV = V + 55.0
        safe_dV = jnp.where(jnp.abs(dV) < 1e-6, 1.0, dV)
        return jnp.where(
            jnp.abs(dV) < 1e-6,
            0.1,
            0.01 * safe_dV / (1.0 - jnp.exp(-safe_dV / 10.0))
        )

    @staticmethod
    def beta_n(V):
        return 0.125 * jnp.exp(-(V + 65.0) / 80.0)

    # ================================================================
    # Steady-state gating
    # ================================================================

    @staticmethod
    def m_inf(V):
        a = HHReference.alpha_m(V)
        return a / (a + HHReference.beta_m(V))

    @staticmethod
    def h_inf(V):
        a = HHReference.alpha_h(V)
        return a / (a + HHReference.beta_h(V))

    @staticmethod
    def n_inf(V):
        a = HHReference.alpha_n(V)
        return a / (a + HHReference.beta_n(V))

    # ================================================================
    # Derivatives (scalar or broadcast-compatible)
    # ================================================================

    def derivatives(self, V, m, h, n, I_ext):
        """
        Compute all four HH derivatives.

        All args can be scalars or arrays of matching shape.

        Returns:
            dVdt, dmdt, dhdt, dndt
        """
        # Ionic currents
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        dVdt = (I_ext - I_Na - I_K - I_L) / self.C_m
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n

        return dVdt, dmdt, dhdt, dndt

    def _derivatives_single(self, state, I_ext):
        """
        Single-point derivative for vmap.

        Args:
            state: (4,) array [V, m, h, n]
            I_ext: scalar

        Returns:
            dydt: (4,) array [dV, dm, dh, dn]
        """
        V, m, h, n = state[0], state[1], state[2], state[3]
        dV, dm, dh, dn = self.derivatives(V, m, h, n, I_ext)
        return jnp.array([dV, dm, dh, dn])

    def derivatives_batch(self, states, I_ext):
        """
        Batch derivative computation via vmap.

        Args:
            states: (N, 4) — each row is [V, m, h, n]
            I_ext:  (N,) — external current per sample

        Returns:
            dydt: (N, 4)
        """
        return jax.vmap(self._derivatives_single)(states, I_ext)

    # ================================================================
    # Utility
    # ================================================================

    def resting_state(self, V_rest=-65.0):
        """Equilibrium state at given resting potential."""
        return jnp.array([
            V_rest,
            self.m_inf(V_rest),
            self.h_inf(V_rest),
            self.n_inf(V_rest),
        ])


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    print("HH Reference Model — Batch Test")
    print("=" * 50)

    hh = HHReference()

    # Single point at rest
    y0 = hh.resting_state()
    dV, dm, dh, dn = hh.derivatives(y0[0], y0[1], y0[2], y0[3], 0.0)
    print(f"Resting state: V={y0[0]:.1f}, m={y0[1]:.4f}, h={y0[2]:.4f}, n={y0[3]:.4f}")
    print(f"Derivatives at rest (should be ~0): dV={dV:.6f}, dm={dm:.6f}, dh={dh:.6f}, dn={dn:.6f}")

    # Batch test
    key = jax.random.PRNGKey(0)
    N = 1000
    states = jax.random.uniform(key, (N, 4),
                                minval=jnp.array([-100.0, 0.0, 0.0, 0.0]),
                                maxval=jnp.array([60.0, 1.0, 1.0, 1.0]))
    I_ext = jax.random.uniform(key, (N,), minval=-10.0, maxval=150.0)

    dydt = hh.derivatives_batch(states, I_ext)
    print(f"\nBatch test ({N} points):")
    print(f"  dydt shape: {dydt.shape}")
    print(f"  dV/dt range: [{dydt[:, 0].min():.1f}, {dydt[:, 0].max():.1f}] mV/ms")
    print(f"  dm/dt range: [{dydt[:, 1].min():.4f}, {dydt[:, 1].max():.4f}]")
    print(f"  dh/dt range: [{dydt[:, 2].min():.4f}, {dydt[:, 2].max():.4f}]")
    print(f"  dn/dt range: [{dydt[:, 3].min():.4f}, {dydt[:, 3].max():.4f}]")

    print("\nHH Reference OK!")
