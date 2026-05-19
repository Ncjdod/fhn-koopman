"""
Vector Field Neural Network

Learns the HH vector field: f(V, m, h, n, I_ext) -> (dV/dt, dm/dt, dh/dt, dn/dt)

Architecture:
  Input normalization (5D -> [-1,1])
  -> Random Fourier Features (captures sharp HH nonlinearities)
  -> MLP via jax.lax.scan (GPU-friendly, no Python loops)
  -> Derivative clipping (prevents integration divergence)

The MLP uses explicit stacked weight matrices with jax.lax.scan over
hidden layers. This produces a compact XLA graph where both forward
and backward passes compile as single loop bodies, avoiding the
graph explosion that causes GPU compilation to hang.

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

    Input:  (V, m, h, n, I_ext) — 5 scalars or batched
    Output: (dV/dt, dm/dt, dh/dt, dn/dt) — 4 scalars or batched

    Architecture uses explicit weight matrices (no eqx.nn.MLP) so that
    hidden layers are stacked into a single array and iterated via
    jax.lax.scan. This compiles to a single loop in XLA instead of
    N unrolled matmul subgraphs.
    """
    B: jnp.ndarray          # (5, n_fourier) — frozen random Fourier frequencies

    # Explicit MLP weights (stacked for lax.scan)
    W_in: jnp.ndarray       # (hidden_dim, 2*n_fourier) — input projection
    b_in: jnp.ndarray       # (hidden_dim,)
    W_hidden: jnp.ndarray   # (n_layers-1, hidden_dim, hidden_dim) — stacked hidden layers
    b_hidden: jnp.ndarray   # (n_layers-1, hidden_dim)

    # Dedicated V head with physics shortcut
    # Input = trunk output (hidden_dim) + physics features (4: Na, K, leak, I_ext)
    _n_physics: int = 4
    W_v1: jnp.ndarray       # (v_head_dim, hidden_dim + _n_physics)
    b_v1: jnp.ndarray       # (v_head_dim,)
    W_v2: jnp.ndarray       # (v_head_dim,)
    b_v2: jnp.ndarray       # scalar

    # Gate heads (stacked — m, h, n share the same head size)
    W_gate1: jnp.ndarray    # (3, head_dim, hidden_dim)
    b_gate1: jnp.ndarray    # (3, head_dim)
    W_gate2: jnp.ndarray    # (3, head_dim)
    b_gate2: jnp.ndarray    # (3,)

    # Normalization parameters
    _v_center: float = -20.0
    _v_scale: float = 80.0
    _i_center: float = 70.0
    _i_scale: float = 80.0

    # Output safety bounds (prevents Euler integration blowup)
    # dV/dt can reach ~18000 mV/ms in extreme state-space corners
    _dV_clip: float = 20000.0
    _dgate_clip: float = 25.0

    def __init__(self, hidden_dim=256, n_layers=4, n_fourier=64, sigma=1.0,
                 head_dim=32, v_head_dim=64, *, key):
        """
        Args:
            hidden_dim:  Width of hidden layers
            n_layers:    Number of hidden layers
            n_fourier:   Number of Fourier frequency pairs (output = 2*n_fourier)
            sigma:       Std of random frequency matrix, or tuple of stds for
                         multi-scale Fourier features (e.g. (1.0, 5.0))
            head_dim:    Width of gate output heads (m, h, n)
            v_head_dim:  Width of voltage output head (larger for complex V dynamics)
            key:         JAX PRNG key
        """
        n_hidden = max(n_layers - 1, 0)
        keys = jax.random.split(key, 6 + n_hidden)
        key_B, key_in = keys[0], keys[1]
        key_v1, key_v2, key_g1, key_g2 = keys[2], keys[3], keys[4], keys[5]
        hidden_keys = keys[6:]

        # Fixed random Fourier frequency matrix: (5, n_fourier)
        if isinstance(sigma, (tuple, list)):
            # Multi-scale: split n_fourier evenly across frequency bands
            n_bands = len(sigma)
            per_band = n_fourier // n_bands
            B_bands = []
            band_keys = jax.random.split(key_B, n_bands)
            for i, s in enumerate(sigma):
                n_this = per_band if i < n_bands - 1 else n_fourier - per_band * (n_bands - 1)
                B_bands.append(jax.random.normal(band_keys[i], (5, n_this)) * s)
            self.B = jnp.concatenate(B_bands, axis=1)
        else:
            self.B = jax.random.normal(key_B, (5, n_fourier)) * sigma

        fourier_dim = 2 * n_fourier  # sin + cos
        init = jax.nn.initializers.lecun_normal()

        # Input layer: fourier_dim -> hidden_dim
        self.W_in = init(key_in, (hidden_dim, fourier_dim))
        self.b_in = jnp.zeros(hidden_dim)

        # Hidden layers: stacked for lax.scan
        if n_hidden > 0:
            W_list = [init(hidden_keys[i], (hidden_dim, hidden_dim))
                      for i in range(n_hidden)]
            self.W_hidden = jnp.stack(W_list)          # (n_hidden, hidden_dim, hidden_dim)
            self.b_hidden = jnp.zeros((n_hidden, hidden_dim))
        else:
            self.W_hidden = jnp.zeros((0, hidden_dim, hidden_dim))
            self.b_hidden = jnp.zeros((0, hidden_dim))

        # Dedicated V head with physics shortcut inputs
        # Receives trunk output + 4 physics features (Na, K, leak, I_ext)
        v_input_dim = hidden_dim + self._n_physics
        self.W_v1 = init(key_v1, (v_head_dim, v_input_dim))
        self.b_v1 = jnp.zeros(v_head_dim)
        self.W_v2 = init(key_v2, (1, v_head_dim))[0]  # (v_head_dim,)
        self.b_v2 = jnp.zeros(())

        # Gate heads: 3 stacked (m, h, n)
        gate_keys1 = jax.random.split(key_g1, 3)
        W_g1_list = [init(gate_keys1[i], (head_dim, hidden_dim)) for i in range(3)]
        self.W_gate1 = jnp.stack(W_g1_list)           # (3, head_dim, hidden_dim)
        self.b_gate1 = jnp.zeros((3, head_dim))

        gate_keys2 = jax.random.split(key_g2, 3)
        W_g2_list = [init(gate_keys2[i], (1, head_dim))[0] for i in range(3)]
        self.W_gate2 = jnp.stack(W_g2_list)           # (3, head_dim)
        self.b_gate2 = jnp.zeros(3)

    @staticmethod
    def _physics_features_single(V, m, h, n, I_ext):
        """Compute HH ionic current structural terms for a single point."""
        I_Na_struct = (m ** 3) * h * (V - 50.0)     # Na current structure
        I_K_struct = (n ** 4) * (V + 77.0)           # K current structure
        I_L_struct = V + 54.4                         # leak current structure
        return jnp.array([I_Na_struct, I_K_struct, I_L_struct, I_ext])

    @staticmethod
    def _physics_features_batch(states, I_ext):
        """Compute HH ionic current structural terms for a batch."""
        V, m, h, n = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        I_Na_struct = (m ** 3) * h * (V - 50.0)
        I_K_struct = (n ** 4) * (V + 77.0)
        I_L_struct = V + 54.4
        return jnp.stack([I_Na_struct, I_K_struct, I_L_struct, I_ext], axis=-1)

    def _forward(self, x_fourier, physics):
        """
        Core MLP forward pass using lax.scan + separate output heads.

        The V head receives trunk output concatenated with physics features
        (ionic current structures), giving it a shortcut to the multiplicative
        terms it otherwise struggles to learn.

        Gate heads (m, h, n) use trunk output only.

        Works for both single-point (D,) and batch (N, D) inputs.
        """
        # Input layer
        x = jnp.tanh(x_fourier @ self.W_in.T + self.b_in)

        # Hidden layers via lax.scan — single loop body in XLA
        def scan_body(x, wb):
            w, b = wb
            return jnp.tanh(x @ w.T + b), None

        x, _ = jax.lax.scan(scan_body, x, (self.W_hidden, self.b_hidden))

        # V head: [trunk_output, physics_features] -> v_head_dim -> 1
        x_v = jnp.concatenate([x, physics], axis=-1)  # (..., hidden_dim + 4)
        h_v = jnp.tanh(x_v @ self.W_v1.T + self.b_v1)        # (..., v_head_dim)
        dV = h_v @ self.W_v2 + self.b_v2                      # (...,)

        # Gate heads (stacked): x -> head_dim -> 1, for m/h/n
        h_g = jnp.tanh(jnp.einsum('...d,ohd->...oh', x, self.W_gate1) + self.b_gate1)
        gates = jnp.einsum('...oh,oh->...o', h_g, self.W_gate2) + self.b_gate2  # (..., 3)

        # Concatenate: (..., 1) + (..., 3) -> (..., 4)
        raw = jnp.concatenate([dV[..., None], gates], axis=-1)

        # Derivative clipping
        clip_vals = jnp.array([self._dV_clip, self._dgate_clip,
                               self._dgate_clip, self._dgate_clip])
        return jnp.clip(raw, -clip_vals, clip_vals)

    def __call__(self, V, m, h, n, I_ext):
        """
        Predict vector field at a single state-space point.

        Returns:
            dydt: (4,) array [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        # Normalize to ~[-1, 1]
        V_norm = (V - self._v_center) / self._v_scale
        m_norm = 2.0 * m - 1.0
        h_norm = 2.0 * h - 1.0
        n_norm = 2.0 * n - 1.0
        I_norm = (I_ext - self._i_center) / self._i_scale
        x_norm = jnp.array([V_norm, m_norm, h_norm, n_norm, I_norm])

        # Fourier embed
        B = jax.lax.stop_gradient(self.B)
        proj = 2.0 * jnp.pi * x_norm @ B
        x_fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])

        # Physics shortcut for V head
        physics = self._physics_features_single(V, m, h, n, I_ext)

        return self._forward(x_fourier, physics)

    def predict_batch(self, states, I_ext):
        """
        Batch prediction via native matrix operations (no vmap, no Python loops).

        Uses the same _forward() as __call__ — matmul broadcasts over batch dim.

        Args:
            states: (N, 4) — each row [V, m, h, n]
            I_ext:  (N,)

        Returns:
            dydt: (N, 4)
        """
        # Batch normalize: (N, 5)
        V_norm = (states[:, 0] - self._v_center) / self._v_scale
        m_norm = 2.0 * states[:, 1] - 1.0
        h_norm = 2.0 * states[:, 2] - 1.0
        n_norm = 2.0 * states[:, 3] - 1.0
        I_norm = (I_ext - self._i_center) / self._i_scale
        x_norm = jnp.stack([V_norm, m_norm, h_norm, n_norm, I_norm], axis=-1)

        # Batch Fourier embedding: (N, 2*n_fourier)
        B = jax.lax.stop_gradient(self.B)
        proj = 2.0 * jnp.pi * (x_norm @ B)
        x_fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

        # Physics shortcut for V head
        physics = self._physics_features_batch(states, I_ext)

        return self._forward(x_fourier, physics)


def create_model(hidden_dim=256, n_layers=4, n_fourier=64, sigma=1.0,
                 head_dim=32, v_head_dim=64, key=None):
    """
    Factory function for VectorFieldNet.

    Args:
        hidden_dim:  MLP width
        n_layers:    MLP depth (hidden layers)
        n_fourier:   Fourier feature pairs (output dim = 2*n_fourier)
        sigma:       Fourier frequency scale
        head_dim:    Gate output head width (m, h, n)
        v_head_dim:  Voltage output head width (larger for complex V dynamics)
        key:         JAX PRNG key (default: PRNGKey(42))

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
        head_dim=head_dim,
        v_head_dim=v_head_dim,
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
    print("VectorFieldNet -- Architecture Test")
    print("=" * 50)

    key = jax.random.PRNGKey(0)
    model = create_model(key=key)

    # Count parameters
    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Parameters: {n_params:,}")

    # Verify structure
    print(f"\nWeight shapes:")
    print(f"  B:        {model.B.shape}")
    print(f"  W_in:     {model.W_in.shape}")
    print(f"  b_in:     {model.b_in.shape}")
    print(f"  W_hidden: {model.W_hidden.shape}")
    print(f"  b_hidden: {model.b_hidden.shape}")
    print(f"  W_v1:     {model.W_v1.shape}")
    print(f"  W_v2:     {model.W_v2.shape}")
    print(f"  W_gate1:  {model.W_gate1.shape}")
    print(f"  W_gate2:  {model.W_gate2.shape}")

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

    # Verify single == batch for same input
    dydt_single = model(states[0, 0], states[0, 1], states[0, 2], states[0, 3], I_ext_batch[0])
    diff = jnp.max(jnp.abs(dydt_single - dydt_batch[0]))
    print(f"  Single vs batch[0] max diff: {diff:.2e}")

    print("\nVectorFieldNet OK!")
