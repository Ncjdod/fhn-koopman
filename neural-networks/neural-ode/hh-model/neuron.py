"""
Composable Single Neuron Module

Wraps a trained HHNeuralODE with a standardized interface for
network composition. Each neuron accepts total input current
(external + synaptic) and exposes its state for other neurons
to read.

Usage:
    from neuron import SingleNeuron
    from HH_NeuralODE import create_model

    model = create_model()  # or load a trained model
    neuron = SingleNeuron(dynamics=model, neuron_id=0)

    # In a network context:
    dy = neuron(t, state, I_total)  # state: [V, m, h, n]
"""

import jax.numpy as jnp
import equinox as eqx

from HodgkinHuxley import HodgkinHuxley
from HH_NeuralODE import HHNeuralODE


class SingleNeuron(eqx.Module):
    """
    A single neuron module with standardized I/O for network composition.

    The neuron wraps a learned HHNeuralODE and provides:
      - Standard call interface: (t, state, I_total) -> d_state/dt
      - Initial state generation from resting potential
      - Voltage accessor for synaptic coupling

    State vector: [V, m, h, n] (4 components)
    """
    dynamics: HHNeuralODE
    neuron_id: int
    state_dim: int = 4  # [V, m, h, n]

    def __call__(self, t, state, I_total):
        """
        Compute state derivatives given total input current.

        Args:
            t:       Current time (scalar, ms)
            state:   This neuron's state [V, m, h, n] (shape: (4,))
            I_total: Total input current = I_ext + I_syn (scalar, pA)

        Returns:
            d_state/dt: [dV/dt, dm/dt, dh/dt, dn/dt] (shape: (4,))
        """
        return self.dynamics(t, state, I_total)

    @staticmethod
    def voltage(state):
        """Extract membrane voltage from state vector."""
        return state[0]

    @staticmethod
    def initial_state(V0=-65.0):
        """
        Return steady-state initial condition at resting potential.

        Args:
            V0: Resting membrane potential (mV)

        Returns:
            y0: [V0, m_inf(V0), h_inf(V0), n_inf(V0)] (shape: (4,))
        """
        hh = HodgkinHuxley()
        return hh.resting_state(V0)


if __name__ == "__main__":
    from HH_NeuralODE import create_model
    import jax

    print("SingleNeuron - Composability Test")
    print("=" * 50)

    key = jax.random.PRNGKey(0)
    model = create_model(key=key)

    neuron = SingleNeuron(dynamics=model, neuron_id=0)

    # Test initial state
    y0 = neuron.initial_state(-65.0)
    print(f"Initial state: V={y0[0]:.1f}, m={y0[1]:.4f}, h={y0[2]:.4f}, n={y0[3]:.4f}")

    # Test forward pass
    t = 0.0
    I_total = 200.0  # pA (external + synaptic)
    dy = neuron(t, y0, I_total)
    print(f"dy/dt: dV={dy[0]:.4f}, dm={dy[1]:.4f}, dh={dy[2]:.4f}, dn={dy[3]:.4f}")

    # Test voltage accessor
    V = neuron.voltage(y0)
    print(f"Voltage: {V:.1f} mV")

    print(f"\nNeuron ID: {neuron.neuron_id}")
    print(f"State dim: {neuron.state_dim}")
    print("\nSingleNeuron OK!")
