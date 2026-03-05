"""
Vector Field Neural Network

Learns the HH vector field: f(V, m, h, n, I_ext) -> (dV/dt, dm/dt, dh/dt, dn/dt)

Architecture:
  Input normalization (5D -> [-1,1]) -> MLP(256 x 4 layers, tanh) -> 4D output

No Fourier features needed — the inputs are already the physical state variables,
not a time coordinate that requires spectral encoding.

No output clipping — the field should be learned cleanly without artificial bounds.

Built on Equinox (pure JAX).
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class VectorFieldNet(eqx.Module):
    """
    Neural network approximation of the HH vector field.

    Input:  (V, m, h, n, I_ext) — 5 scalars
    Output: (dV/dt, dm/dt, dh/dt, dn/dt) — 4 scalars

    Normalization constants (fixed, not trainable):
        V:     [-100, 60]  -> [-1, 1]   center=-20, half_range=80
        m,h,n: [0, 1]      -> [-1, 1]
        I_ext: [-10, 150]  -> [-1, 1]   center=70, half_range=80
    """
    mlp: eqx.nn.MLP

    # Normalization parameters (not trainable, just constants)
    _v_center: float = -20.0
    _v_scale: float = 80.0
    _i_center: float = 70.0
    _i_scale: float = 80.0

    def __init__(self, hidden_dim=256, n_layers=4, *, key):
        """
        Args:
            hidden_dim: Width of hidden layers
            n_layers:   Number of hidden layers
            key:        JAX PRNG key
        """
        self.mlp = eqx.nn.MLP(
            in_size=5,
            out_size=4,
            width_size=hidden_dim,
            depth=n_layers,
            activation=jnp.tanh,
            key=key,
        )

    def normalize(self, V, m, h, n, I_ext):
        """Normalize inputs to approximately [-1, 1]."""
        V_norm = (V - self._v_center) / self._v_scale
        m_norm = 2.0 * m - 1.0
        h_norm = 2.0 * h - 1.0
        n_norm = 2.0 * n - 1.0
        I_norm = (I_ext - self._i_center) / self._i_scale
        return jnp.array([V_norm, m_norm, h_norm, n_norm, I_norm])

    def __call__(self, V, m, h, n, I_ext):
        """
        Predict vector field at a single state-space point.

        Args:
            V:     Membrane voltage (mV), scalar
            m:     Na+ activation gating, scalar
            h:     Na+ inactivation gating, scalar
            n:     K+ activation gating, scalar
            I_ext: External current (uA/cm^2), scalar

        Returns:
            dydt: (4,) array [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        x = self.normalize(V, m, h, n, I_ext)
        return self.mlp(x)

    def predict_batch(self, states, I_ext):
        """
        Batch prediction via vmap.

        Args:
            states: (N, 4) — each row [V, m, h, n]
            I_ext:  (N,)

        Returns:
            dydt: (N, 4)
        """
        def single(state, I):
            return self(state[0], state[1], state[2], state[3], I)
        return jax.vmap(single)(states, I_ext)


def create_model(hidden_dim=256, n_layers=4, key=None):
    """
    Factory function for VectorFieldNet.

    Args:
        hidden_dim: MLP width
        n_layers:   MLP depth (hidden layers)
        key:        JAX PRNG key (default: PRNGKey(42))

    Returns:
        VectorFieldNet instance
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    return VectorFieldNet(hidden_dim=hidden_dim, n_layers=n_layers, key=key)


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    print("VectorFieldNet — Architecture Test")
    print("=" * 50)

    key = jax.random.PRNGKey(0)
    model = create_model(key=key)

    # Count parameters
    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Parameters: {n_params:,}")

    # Single point
    V, m, h, n, I_ext = -65.0, 0.05, 0.6, 0.32, 10.0
    dydt = model(V, m, h, n, I_ext)
    print(f"\nSingle point test:")
    print(f"  Input:  V={V}, m={m}, h={h}, n={n}, I_ext={I_ext}")
    print(f"  Output: dV={dydt[0]:.4f}, dm={dydt[1]:.4f}, dh={dydt[2]:.4f}, dn={dydt[3]:.4f}")

    # Batch test
    N = 1000
    states = jax.random.uniform(key, (N, 4),
                                minval=jnp.array([-100.0, 0.0, 0.0, 0.0]),
                                maxval=jnp.array([60.0, 1.0, 1.0, 1.0]))
    I_ext_batch = jax.random.uniform(key, (N,), minval=-10.0, maxval=150.0)

    dydt_batch = model.predict_batch(states, I_ext_batch)
    print(f"\nBatch test ({N} points):")
    print(f"  Output shape: {dydt_batch.shape}")
    print(f"  dV/dt range: [{dydt_batch[:, 0].min():.4f}, {dydt_batch[:, 0].max():.4f}]")

    print("\nVectorFieldNet OK!")
