"""
HH Neural ODE Model

Architecture:
  - Input normalization (scales heterogeneous units to ~[-1, 1])
  - Fixed Fourier Features (non-trainable) for spectral learning
  - 4-layer MLP (128 neurons each) with tanh activation
  - 4D output: [dV/dt, dm/dt, dh/dt, dn/dt]
  - Diffrax adaptive integration

Built on: Equinox + Diffrax + Optax (pure JAX stack)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from functools import partial


# ============================================================
# Fixed Fourier Features
# ============================================================
class FourierFeatures(eqx.Module):
    """
    Fixed (non-trainable) random Fourier feature encoding.

    Maps input x -> [sin(2*pi*B*x), cos(2*pi*B*x)]
    where B is a fixed random matrix sampled from N(0, sigma^2).

    This helps the network learn multi-scale dynamics by providing
    a rich spectral basis as input features.
    """
    B: jnp.ndarray  # Fixed frequency matrix (non-trainable)

    def __init__(self, input_dim, n_features, sigma=1.0, *, key):
        """
        Args:
            input_dim:  Dimension of input (6 for [t, V, m, h, n, I_ext])
            n_features: Number of Fourier basis functions
            sigma:      Std dev of frequency sampling (controls scale)
            key:        JAX PRNG key
        """
        self.B = jax.random.normal(key, (input_dim, n_features)) * sigma

    def __call__(self, x):
        """
        Args:
            x: Input array of shape (input_dim,)
        Returns:
            Fourier features of shape (2 * n_features,)
        """
        projection = 2.0 * jnp.pi * x @ self.B  # (n_features,)
        return jnp.concatenate([jnp.sin(projection), jnp.cos(projection)])


# ============================================================
# HH Neural ODE Model (4D State)
# ============================================================
class HHNeuralODE(eqx.Module):
    """
    Neural ODE for learning full Hodgkin-Huxley dynamics.

    Architecture:
        Input: [t, V, m, h, n, I_ext] -> normalize -> FourierFeatures
               -> 4x(Linear(128) + tanh) -> Linear(4)
               -> [dV/dt, dm/dt, dh/dt, dn/dt]

    The model learns the full 4D dynamics:
        dy/dt = f_net(t, V, m, h, n, I_ext)
    where y = [V, m, h, n].
    """
    fourier: FourierFeatures
    mlp: eqx.nn.MLP

    def __init__(self, n_fourier=32, sigma=1.0, *, key):
        """
        Args:
            n_fourier: Number of Fourier basis functions (output dim = 2*n_fourier)
            sigma:     Fourier frequency scale
            key:       JAX PRNG key
        """
        keys = jax.random.split(key, 2)

        input_dim = 6 
        fourier_out_dim = 2 * n_fourier  
        hidden_dim = 128

        # Fixed Fourier features
        self.fourier = FourierFeatures(input_dim, n_fourier, sigma=sigma, key=keys[0])

        # 4 hidden layers, 128 neurons each
        self.mlp = eqx.nn.MLP(
        in_size=fourier_out_dim,
        out_size=4,
        width_size=128,
        depth=4,
        activation=jnp.tanh,
        key=keys[1]
        )

    @staticmethod
    def normalize_inputs(t, V, m, h, n, I_ext):
        """
        Normalize inputs to ~[-1, 1] range for stable Fourier encoding.

        Input scales:
            t:     [0, 55] ms
            V:     [-80, 40] mV
            m,h,n: [0, 1]
            I_ext: [0, ~300] pA
        """
        t_norm = t / 27.5 - 1.0        # [0, 55] -> [-1, 1]
        V_norm = (V + 20.0) / 60.0     # [-80, 40] -> [-1.67, 0.33] (centered)
        m_norm = m * 2.0 - 1.0         # [0, 1] -> [-1, 1]
        h_norm = h * 2.0 - 1.0         # [0, 1] -> [-1, 1]
        n_norm = n * 2.0 - 1.0         # [0, 1] -> [-1, 1]
        I_norm = I_ext / 150.0 - 1.0   # [0, 300] -> [-1, 1]
        return jnp.array([t_norm, V_norm, m_norm, h_norm, n_norm, I_norm])

    def __call__(self, t, y, I_ext):
        """
        Compute dy/dt for the full 4D HH state.

        Args:
            t:     Current time (scalar, ms)
            y:     Current state [V, m, h, n] (shape: (4,))
            I_ext: External current at time t (scalar, pA)

        Returns:
            dy/dt: Time derivatives [dV/dt, dm/dt, dh/dt, dn/dt] (shape: (4,))
        """
        # Normalize inputs to ~[-1, 1]
        x = self.normalize_inputs(t, y[0], y[1], y[2], y[3], I_ext)

        # Fourier encoding
        x = self.fourier(x)

        out = self.mlp(x)

        dVdt = out[0:1]    
        out_gates = out[1:4] 
        y_gates = y[1:4]      

        dgates_dt = jnp.where(out_gates > 0, 
                      out_gates * (1.0 - y_gates), 
                      out_gates * y_gates)

        return jnp.concatenate([dVdt, dgates_dt])


# ============================================================
# ODE Integration (Diffrax)
# ============================================================
def make_diffrax_term(I_ext_fn):
    """
    Create a diffrax ODETerm that reads the model from args.

    The model is passed via diffeqsolve's `args` parameter rather than
    being captured in a closure. This is required for BacksolveAdjoint
    (continuous adjoint method), which uses a custom VJP rule that can
    only differentiate with respect to explicit args, not closed-over values.

    Args:
        I_ext_fn:  Function t -> I_ext (external current at time t).
                   This is NOT differentiated, so closure capture is fine.

    Returns:
        diffrax.ODETerm
    """
    def vector_field(t, y, args):
        model = args
        I_ext = I_ext_fn(t)
        return model(t, y, I_ext)

    return diffrax.ODETerm(vector_field)


def integrate(model, y0, t_span, I_ext_fn, dt0=0.01, solver=None,
              rtol=1e-3, atol=1e-5, max_steps=16384, adjoint=None):
    """
    Integrate the Neural ODE forward in time.

    Args:
        model:     HHNeuralODE instance
        y0:        Initial state [V0, m0, h0, n0] (shape: (4,))
        t_span:    Array of output times (shape: (n_steps,))
        I_ext_fn:  Function t -> I_ext
        dt0:       Initial step size
        solver:    Diffrax solver (default: Tsit5)
        rtol, atol: Tolerances for adaptive stepping
        max_steps: Maximum solver steps (default 16384, use 4096 for segments)
        adjoint:   Diffrax adjoint method for backpropagation through the solver.
                   None defaults to RecursiveCheckpointAdjoint (discretise-then-optimise).
                   Use diffrax.BacksolveAdjoint() for continuous adjoint (memory-efficient).

    Returns:
        ys: Trajectory of shape (n_steps, 4)
    """
    if solver is None:
        solver = diffrax.Tsit5()

    if adjoint is None:
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    term = make_diffrax_term(I_ext_fn)

    saveat = diffrax.SaveAt(ts=t_span)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=dt0,
        y0=y0,
        args=model,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        max_steps=max_steps,
        throw=False,
    )

    return sol.ys  # (n_steps, 4)


# ============================================================
# Model Factory
# ============================================================
def create_model(key=None, n_fourier=32, sigma=1.0):
    """
    Create a new HHNeuralODE model.

    Args:
        key:       JAX PRNG key (default: random seed 42)
        n_fourier: Number of Fourier basis functions
        sigma:     Fourier frequency scale

    Returns:
        model: HHNeuralODE instance
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    model = HHNeuralODE(n_fourier=n_fourier, sigma=sigma, key=key)
    return model


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("HH Neural ODE - Architecture Test (4D State)")
    print("=" * 50)

    # Create model
    key = jax.random.PRNGKey(0)
    model = create_model(key=key, n_fourier=32, sigma=1.0)

    # Count parameters
    params = eqx.filter(model, eqx.is_array)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Total parameters: {n_params}")

    # Test forward pass
    t = 0.0
    y = jnp.array([-65.0, 0.05, 0.6, 0.32])  # [V, m, h, n] at rest
    I_ext = 200.0  # pA

    dy = model(t, y, I_ext)
    print(f"Input:  t={t}, V={y[0]:.1f}, m={y[1]:.3f}, h={y[2]:.3f}, n={y[3]:.3f}, I_ext={I_ext}")
    print(f"Output: dV/dt={dy[0]:.4f}, dm/dt={dy[1]:.4f}, dh/dt={dy[2]:.4f}, dn/dt={dy[3]:.4f}")

    # Test integration
    from HodgkinHuxley import HodgkinHuxley
    hh = HodgkinHuxley()

    t_span = jnp.linspace(0.0, 10.0, 100)  # 10ms
    y0 = hh.resting_state(-65.0)  # [V, m_inf, h_inf, n_inf]
    I_ext_fn = lambda t: 200.0  # Constant stimulus

    ys = integrate(model, y0, t_span, I_ext_fn)
    print(f"\nIntegration test (10ms, 200 pA):")
    print(f"  y0: V={y0[0]:.1f}, m={y0[1]:.4f}, h={y0[2]:.4f}, n={y0[3]:.4f}")
    print(f"  V(0)  = {ys[0, 0]:.2f} mV")
    print(f"  V(10) = {ys[-1, 0]:.2f} mV")
    print(f"  V range: [{ys[:, 0].min():.2f}, {ys[:, 0].max():.2f}]")
    print(f"  m range: [{ys[:, 1].min():.4f}, {ys[:, 1].max():.4f}]")
    print(f"  h range: [{ys[:, 2].min():.4f}, {ys[:, 2].max():.4f}]")
    print(f"  n range: [{ys[:, 3].min():.4f}, {ys[:, 3].max():.4f}]")

    print("\nArchitecture OK!")
