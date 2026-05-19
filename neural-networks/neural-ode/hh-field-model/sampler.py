"""
State-Space Sampling for Phase 1 Training

Generates random (state, I_ext) points covering the HH state space.
Three sampling strategies:
  1. Uniform:        covers full hypercube (handles extremes)
  2. Physiological:  concentrated near the HH manifold (where dynamics matter)
  3. Mixed:          blend of both (default for training)

All samples are generated online — new batch each epoch, no fixed dataset.
"""

import jax
import jax.numpy as jnp
from functools import partial

from hh_reference import HHReference


class StateSpaceSampler:
    """
    Generates training samples in HH state space.

    State: [V, m, h, n]   (4D)
    Input: I_ext           (scalar per sample)

    Ranges:
        V:     [-100, 60] mV
        m,h,n: [0, 1]
        I_ext: [-10, 150] uA/cm^2
    """

    def __init__(self,
                 V_range=(-100.0, 60.0),
                 m_range=(0.0, 1.0),
                 h_range=(0.0, 1.0),
                 n_range=(0.0, 1.0),
                 I_ext_range=(-10.0, 150.0),
                 V_mean=-65.0,
                 V_std=30.0,
                 gate_std=0.15):
        self.V_range = V_range
        self.m_range = m_range
        self.h_range = h_range
        self.n_range = n_range
        self.I_ext_range = I_ext_range
        self.V_mean = V_mean
        self.V_std = V_std
        self.gate_std = gate_std
        self.hh = HHReference()

    def uniform_sample(self, key, n_samples):
        """
        Uniform random sampling across the full state-space hypercube.

        Args:
            key:       JAX PRNG key
            n_samples: Number of points to generate

        Returns:
            states: (N, 4) — [V, m, h, n]
            I_ext:  (N,)
        """
        keys = jax.random.split(key, 5)

        V = jax.random.uniform(keys[0], (n_samples,),
                               minval=self.V_range[0], maxval=self.V_range[1])
        m = jax.random.uniform(keys[1], (n_samples,),
                               minval=self.m_range[0], maxval=self.m_range[1])
        h = jax.random.uniform(keys[2], (n_samples,),
                               minval=self.h_range[0], maxval=self.h_range[1])
        n = jax.random.uniform(keys[3], (n_samples,),
                               minval=self.n_range[0], maxval=self.n_range[1])
        I_ext = jax.random.uniform(keys[4], (n_samples,),
                                   minval=self.I_ext_range[0],
                                   maxval=self.I_ext_range[1])

        states = jnp.stack([V, m, h, n], axis=-1)
        return states, I_ext

    def physiological_sample(self, key, n_samples):
        """
        Importance sampling near the physiological manifold.

        Strategy:
          1. Sample V from N(V_mean, V_std^2), clamp to V_range
          2. Compute steady-state gating: m_inf(V), h_inf(V), n_inf(V)
          3. Add Gaussian noise: gate ~ N(gate_inf, gate_std^2), clamp to [0, 1]
          4. I_ext sampled uniformly

        This concentrates samples where the dynamics are biologically relevant
        while still providing some off-manifold coverage via the noise.

        Args:
            key:       JAX PRNG key
            n_samples: Number of points

        Returns:
            states: (N, 4)
            I_ext:  (N,)
        """
        keys = jax.random.split(key, 5)

        # Voltage: Gaussian centered at rest
        V = jax.random.normal(keys[0], (n_samples,)) * self.V_std + self.V_mean
        V = jnp.clip(V, self.V_range[0], self.V_range[1])

        # Gating: noisy steady-state
        m_inf = self.hh.m_inf(V)
        h_inf = self.hh.h_inf(V)
        n_inf = self.hh.n_inf(V)

        m = m_inf + jax.random.normal(keys[1], (n_samples,)) * self.gate_std
        h = h_inf + jax.random.normal(keys[2], (n_samples,)) * self.gate_std
        n = n_inf + jax.random.normal(keys[3], (n_samples,)) * self.gate_std

        m = jnp.clip(m, 0.0, 1.0)
        h = jnp.clip(h, 0.0, 1.0)
        n = jnp.clip(n, 0.0, 1.0)

        I_ext = jax.random.uniform(keys[4], (n_samples,),
                                   minval=self.I_ext_range[0],
                                   maxval=self.I_ext_range[1])

        states = jnp.stack([V, m, h, n], axis=-1)
        return states, I_ext

    def mixed_sample(self, key, n_samples, phys_fraction=0.7):
        """
        Blend of physiological and uniform sampling.

        Default: 70% physiological, 30% uniform.
        This ensures:
          - Good coverage of the biologically relevant manifold
          - Robustness to off-manifold states (prevents extrapolation failures)

        Args:
            key:           JAX PRNG key
            n_samples:     Total number of points
            phys_fraction: Fraction of physiological samples

        Returns:
            states: (N, 4)
            I_ext:  (N,)
        """
        n_phys = int(n_samples * phys_fraction)
        n_unif = n_samples - n_phys

        key1, key2 = jax.random.split(key)

        states_phys, I_phys = self.physiological_sample(key1, n_phys)
        states_unif, I_unif = self.uniform_sample(key2, n_unif)

        states = jnp.concatenate([states_phys, states_unif], axis=0)
        I_ext = jnp.concatenate([I_phys, I_unif], axis=0)

        return states, I_ext


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    print("State Space Sampler — Test")
    print("=" * 50)

    sampler = StateSpaceSampler()
    key = jax.random.PRNGKey(0)

    for name, method in [("uniform", sampler.uniform_sample),
                         ("physiological", sampler.physiological_sample),
                         ("mixed (70/30)", partial(sampler.mixed_sample, phys_fraction=0.7))]:
        states, I_ext = method(key, 5000)
        print(f"\n{name} ({states.shape[0]} samples):")
        print(f"  V:     [{states[:, 0].min():.1f}, {states[:, 0].max():.1f}], "
              f"mean={states[:, 0].mean():.1f}")
        print(f"  m:     [{states[:, 1].min():.4f}, {states[:, 1].max():.4f}], "
              f"mean={states[:, 1].mean():.4f}")
        print(f"  h:     [{states[:, 2].min():.4f}, {states[:, 2].max():.4f}], "
              f"mean={states[:, 2].mean():.4f}")
        print(f"  n:     [{states[:, 3].min():.4f}, {states[:, 3].max():.4f}], "
              f"mean={states[:, 3].mean():.4f}")
        print(f"  I_ext: [{I_ext.min():.1f}, {I_ext.max():.1f}]")

    print("\nSampler OK!")
