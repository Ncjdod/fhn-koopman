"""
Learnable Latent State for Phase 2 Boundary Condition

Allen Brain data provides V(t) and I_ext(t), but not the gating variables
m(t), h(t), n(t). These are treated as learnable parameters that get
jointly optimized with the model weights during Phase 2.

Parameterization:
    m(t) = sigmoid(m_raw(t))   -> guarantees m in (0, 1)
    h(t) = sigmoid(h_raw(t))
    n(t) = sigmoid(n_raw(t))

Initialization:
    m_raw = logit(m_inf(V_obs))   (HH steady-state as starting point)

ConversionFactor:
    Trainable pA -> uA/cm^2 conversion for matching Allen data to HH units.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from hh_reference import HHReference


class LatentGatingState(eqx.Module):
    """
    Learnable gating variables along an observed voltage trajectory.

    The raw parameters (m_raw, h_raw, n_raw) are unconstrained reals.
    The actual gating values are obtained via sigmoid, ensuring (0, 1).

    Attributes:
        m_raw: (T,) unconstrained parameters for Na+ activation
        h_raw: (T,) unconstrained parameters for Na+ inactivation
        n_raw: (T,) unconstrained parameters for K+ activation
    """
    m_raw: jnp.ndarray
    h_raw: jnp.ndarray
    n_raw: jnp.ndarray

    def __init__(self, V_obs):
        """
        Initialize latent gating from HH steady-state at observed voltages.

        Args:
            V_obs: (T,) observed voltage in mV (JAX array)
        """
        hh = HHReference()
        eps = 1e-6

        # Compute steady-state gating
        m0 = jnp.clip(hh.m_inf(V_obs), eps, 1.0 - eps)
        h0 = jnp.clip(hh.h_inf(V_obs), eps, 1.0 - eps)
        n0 = jnp.clip(hh.n_inf(V_obs), eps, 1.0 - eps)

        # Logit transform: x -> log(x / (1 - x))
        self.m_raw = jnp.log(m0 / (1.0 - m0))
        self.h_raw = jnp.log(h0 / (1.0 - h0))
        self.n_raw = jnp.log(n0 / (1.0 - n0))

    @property
    def m(self):
        """Na+ activation gating, (T,), in (0, 1)."""
        return jax.nn.sigmoid(self.m_raw)

    @property
    def h(self):
        """Na+ inactivation gating, (T,), in (0, 1)."""
        return jax.nn.sigmoid(self.h_raw)

    @property
    def n(self):
        """K+ activation gating, (T,), in (0, 1)."""
        return jax.nn.sigmoid(self.n_raw)

    def states(self, V_obs):
        """
        Build full state vectors [V, m, h, n] at each time point.

        Args:
            V_obs: (T,) observed voltage

        Returns:
            states: (T, 4) — each row is [V, m, h, n]
        """
        return jnp.stack([V_obs, self.m, self.h, self.n], axis=-1)


class ConversionFactor(eqx.Module):
    """
    Trainable pA -> uA/cm^2 conversion factor.

    The conversion is parameterized in log-space to ensure positivity:
        factor = exp(log_factor)
        I_hh = I_pA * factor

    Physical meaning:
        factor = 1e-6 / membrane_area_cm2
        (1 pA = 1e-12 A, convert to uA/cm^2 = 1e-6 A/cm^2)

    Default initialization:
        membrane_area = 2e-5 cm^2 (~2000 um^2, typical cortical soma)
        -> factor = 1e-6 / 2e-5 = 0.05
    """
    log_factor: jnp.ndarray  # scalar

    def __init__(self, membrane_area_cm2=2e-5):
        """
        Args:
            membrane_area_cm2: Initial membrane area estimate (cm^2)
        """
        factor = 1e-6 / membrane_area_cm2
        self.log_factor = jnp.log(jnp.array(factor))

    @property
    def factor(self):
        """Conversion factor (always positive)."""
        return jnp.exp(self.log_factor)

    @property
    def membrane_area_cm2(self):
        """Inferred membrane area in cm^2."""
        return 1e-6 / self.factor

    @property
    def membrane_area_um2(self):
        """Inferred membrane area in um^2."""
        return self.membrane_area_cm2 * 1e8

    def convert(self, I_pA):
        """
        Convert current from pA to uA/cm^2.

        Args:
            I_pA: Current in picoamperes (scalar or array)

        Returns:
            I_hh: Current in uA/cm^2
        """
        return I_pA * self.factor


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    print("Latent State — Test")
    print("=" * 50)

    # Fake voltage trajectory
    T = 100
    V_obs = jnp.linspace(-65.0, 30.0, T)

    # Create latent state
    latent = LatentGatingState(V_obs)
    print(f"Latent shapes: m={latent.m.shape}, h={latent.h.shape}, n={latent.n.shape}")
    print(f"m range: [{latent.m.min():.4f}, {latent.m.max():.4f}]")
    print(f"h range: [{latent.h.min():.4f}, {latent.h.max():.4f}]")
    print(f"n range: [{latent.n.min():.4f}, {latent.n.max():.4f}]")

    # Full states
    states = latent.states(V_obs)
    print(f"States shape: {states.shape}")

    # Conversion factor
    conv = ConversionFactor(membrane_area_cm2=2e-5)
    print(f"\nConversion factor: {conv.factor:.6f}")
    print(f"Membrane area: {conv.membrane_area_um2:.0f} um^2")
    print(f"200 pA -> {conv.convert(200.0):.4f} uA/cm^2")

    # Check trainable parameters
    n_latent_params = sum(p.size for p in jax.tree.leaves(
        eqx.filter(latent, eqx.is_array)
    ))
    n_conv_params = sum(p.size for p in jax.tree.leaves(
        eqx.filter(conv, eqx.is_array)
    ))
    print(f"\nTrainable params: latent={n_latent_params}, conversion={n_conv_params}")

    print("\nLatent State OK!")
