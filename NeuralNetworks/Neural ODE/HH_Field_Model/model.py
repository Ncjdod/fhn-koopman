"""
Vector Field Neural Network

Learns the HH vector field: f(V, m, h, n, I_ext) -> (dV/dt, dm/dt, dh/dt, dn/dt)

Architecture:
  Input normalization (5D -> [-1,1])
  -> Random Fourier Features (captures sharp HH nonlinearities)
  -> MLP(256 x 4 layers, tanh)
  -> Derivative clipping (prevents integration divergence)

Fourier features overcome the spectral bias of standard MLPs, enabling
representation of the sharp exponential rate functions in HH dynamics.

Gate safety is enforced at INTEGRATION TIME (clip gates to [0,1] after
each Euler step) rather than in the forward pass, to avoid vanishing
gradients during training.

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
    """
    mlp: eqx.nn.MLP
    B: jnp.ndarray          # Fixed random Fourier frequency matrix (not trained)

    # Normalization parameters
    _v_center: float = -20.0
    _v_scale: float = 80.0
    _i_center: float = 70.0
    _i_scale: float = 80.0

    # Output safety bounds (prevents Euler integration blowup)
    _dV_clip: float = 500.0
    _dgate_clip: float = 25.0

    def __init__(self, hidden_dim=256, n_layers=4, n_fourier=64, sigma=1.0, *, key):
        """
        Args:
            hidden_dim:  Width of hidden layers
            n_layers:    Number of hidden layers
            n_fourier:   Number of Fourier frequency pairs (output = 2*n_fourier)
            sigma:       Std of random frequency matrix (higher = sharper features)
            key:         JAX PRNG key
        """
        key_B, key_mlp = jax.random.split(key)

        # Fixed random Fourier frequency matrix: (5, n_fourier)
        self.B = jax.random.normal(key_B, (5, n_fourier)) * sigma

        fourier_dim = 2 * n_fourier  # sin + cos
        self.mlp = eqx.nn.MLP(
            in_size=fourier_dim,
            out_size=4,
            width_size=hidden_dim,
            depth=n_layers,
            activation=jnp.tanh,
            key=key_mlp,
        )

    def normalize(self, V, m, h, n, I_ext):
        """Normalize inputs to approximately [-1, 1]."""
        V_norm = (V - self._v_center) / self._v_scale
        m_norm = 2.0 * m - 1.0
        h_norm = 2.0 * h - 1.0
        n_norm = 2.0 * n - 1.0
        I_norm = (I_ext - self._i_center) / self._i_scale
        return jnp.array([V_norm, m_norm, h_norm, n_norm, I_norm])

    def fourier_embed(self, x_norm):
        """
        Random Fourier features: project input through fixed random matrix
        then apply sin/cos to capture high-frequency content.

        gamma(x) = [sin(2*pi*x@B), cos(2*pi*x@B)]
        """
        B = jax.lax.stop_gradient(self.B)  # Never train the frequency matrix
        proj = 2.0 * jnp.pi * x_norm @ B   # (n_fourier,)
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])  # (2*n_fourier,)

    def __call__(self, V, m, h, n, I_ext):
        """
        Predict vector field at a single state-space point.

        Returns:
            dydt: (4,) array [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        x_norm = self.normalize(V, m, h, n, I_ext)
        x_fourier = self.fourier_embed(x_norm)
        raw = self.mlp(x_fourier)

        # Clip to prevent integration divergence
        dV = jnp.clip(raw[0], -self._dV_clip, self._dV_clip)
        dm = jnp.clip(raw[1], -self._dgate_clip, self._dgate_clip)
        dh = jnp.clip(raw[2], -self._dgate_clip, self._dgate_clip)
        dn = jnp.clip(raw[3], -self._dgate_clip, self._dgate_clip)

        return jnp.array([dV, dm, dh, dn])

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


def create_model(hidden_dim=256, n_layers=4, n_fourier=64, sigma=1.0, key=None):
    """
    Factory function for VectorFieldNet.

    Args:
        hidden_dim: MLP width
        n_layers:   MLP depth (hidden layers)
        n_fourier:  Fourier feature pairs (output dim = 2*n_fourier)
        sigma:      Fourier frequency scale
        key:        JAX PRNG key (default: PRNGKey(42))

    Returns:
        VectorFieldNet instance
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    return VectorFieldNet(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_fourier=n_fourier,
        sigma=sigma,
        key=key,
    )


# ================================================================
# Safe integration utilities
# ================================================================

def safe_euler_step(model, y, I_ext, dt):
    """
    One Euler step with gate clipping for integration safety.

    Clips gating variables to [0, 1] after each step to prevent escape.
    Use this instead of raw Euler during evaluation/integration.
    """
    dy = model(y[0], y[1], y[2], y[3], I_ext)
    y_new = y + dt * dy
    # Clip gates to [0, 1] — structural safety during integration
    y_new = y_new.at[1].set(jnp.clip(y_new[1], 0.0, 1.0))
    y_new = y_new.at[2].set(jnp.clip(y_new[2], 0.0, 1.0))
    y_new = y_new.at[3].set(jnp.clip(y_new[3], 0.0, 1.0))
    return y_new


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

    # Safe integration test
    from hh_reference import HHReference
    hh = HHReference()
    y0 = hh.resting_state(-65.0)
    y = y0
    for _ in range(100):
        y = safe_euler_step(model, y, 10.0, 0.01)
    print(f"\n100-step safe integration (dt=0.01, I=10):")
    print(f"  V={float(y[0]):.1f}, m={float(y[1]):.4f}, h={float(y[2]):.4f}, n={float(y[3]):.4f}")
    print(f"  Gates in [0,1]: {all(0 <= float(y[i]) <= 1 for i in [1,2,3])}")

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
