"""
Conductance-Based Synapse Model

Models synaptic transmission between neurons using a first-order
kinetic scheme. When the presynaptic neuron fires (V_pre crosses
threshold), neurotransmitter is released, opening postsynaptic
conductances that drive current into the target neuron.

Synapse types:
  - Excitatory (AMPA-like): E_syn = 0 mV, fast kinetics
  - Inhibitory (GABA_A-like): E_syn = -80 mV, slower kinetics

Each synapse adds one state variable 's' (synaptic gating, [0,1])
to the ODE system.

Reference: Destexhe, Mainen & Sejnowski (1994)
"""

import jax.numpy as jnp


class Synapse:
    """
    Conductance-based synapse with fixed parameters.

    Dynamics:
        ds/dt = T(V_pre) * (1 - s) / tau_rise  -  s / tau_decay

    where T(V_pre) is a sigmoid transmitter release function.

    Synaptic current into postsynaptic neuron:
        I_syn = g_max * s * (V_post - E_syn)

    Convention: positive I_syn is depolarizing (excitatory for E_syn > V_rest).
    """

    def __init__(self, g_max=0.1, E_syn=0.0, tau_rise=0.5, tau_decay=5.0,
                 V_thresh=-20.0, slope=2.0):
        """
        Args:
            g_max:    Maximum synaptic conductance (pA/mV, or nS)
            E_syn:    Synaptic reversal potential (mV)
                      0 mV for excitatory (AMPA), -80 mV for inhibitory (GABA_A)
            tau_rise:  Rise time constant (ms)
            tau_decay: Decay time constant (ms)
            V_thresh:  Presynaptic voltage threshold for transmitter release (mV)
            slope:     Steepness of the sigmoid activation (mV)
        """
        self.g_max = g_max
        self.E_syn = E_syn
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.V_thresh = V_thresh
        self.slope = slope

    def transmitter_release(self, V_pre):
        """
        Sigmoid transmitter release as a function of presynaptic voltage.

        T(V) = 1 / (1 + exp(-(V - V_thresh) / slope))

        Smooth approximation of threshold-crossing behavior.

        Args:
            V_pre: Presynaptic membrane voltage (mV)

        Returns:
            T: Transmitter concentration [0, 1]
        """
        return 1.0 / (1.0 + jnp.exp(-(V_pre - self.V_thresh) / self.slope))

    def ds_dt(self, s, V_pre):
        """
        Synaptic gating variable dynamics.

        ds/dt = T(V_pre) * (1 - s) / tau_rise  -  s / tau_decay

        Args:
            s:     Current synaptic gating variable [0, 1]
            V_pre: Presynaptic voltage (mV)

        Returns:
            ds/dt: Time derivative of synaptic gating (scalar)
        """
        T = self.transmitter_release(V_pre)
        return T * (1.0 - s) / self.tau_rise - s / self.tau_decay

    def current(self, s, V_post):
        """
        Synaptic current into the postsynaptic neuron.

        I_syn = g_max * s * (V_post - E_syn)

        Note: For excitatory synapses (E_syn=0), when V_post ~ -65 mV,
        I_syn < 0 (inward current), which depolarizes the neuron when
        subtracted in the HH convention I_ext - I_ion. However, since
        the Neural ODE learned the Allen Brain convention where I_ext
        is injected current (positive = depolarizing), we negate:

        I_syn_injected = -g_max * s * (V_post - E_syn)

        This makes excitatory synapses produce positive (depolarizing) current.

        Args:
            s:      Synaptic gating variable [0, 1]
            V_post: Postsynaptic membrane voltage (mV)

        Returns:
            I_syn: Synaptic current (pA, positive = depolarizing)
        """
        return -self.g_max * s * (V_post - self.E_syn)

    def initial_state(self):
        """Return resting synaptic state (closed)."""
        return 0.0


def excitatory_synapse(g_max=0.5):
    """Create an AMPA-like excitatory synapse."""
    return Synapse(
        g_max=g_max,
        E_syn=0.0,        # AMPA reversal
        tau_rise=0.5,      # Fast rise (ms)
        tau_decay=5.0,     # Moderate decay (ms)
        V_thresh=-20.0,
        slope=2.0,
    )


def inhibitory_synapse(g_max=1.0):
    """Create a GABA_A-like inhibitory synapse."""
    return Synapse(
        g_max=g_max,
        E_syn=-80.0,       # GABA_A reversal
        tau_rise=1.0,       # Slower rise (ms)
        tau_decay=10.0,     # Slow decay (ms)
        V_thresh=-20.0,
        slope=2.0,
    )


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("Synapse Model - Test")
    print("=" * 50)

    syn = excitatory_synapse(g_max=0.5)

    # Test transmitter release
    V_rest = -65.0
    V_spike = 30.0
    print(f"T(V_rest={V_rest}) = {syn.transmitter_release(V_rest):.6f}")
    print(f"T(V_spike={V_spike}) = {syn.transmitter_release(V_spike):.6f}")

    # Test dynamics
    s = 0.0
    print(f"\nAt s=0, V_pre=30 mV:")
    print(f"  ds/dt = {syn.ds_dt(s, V_spike):.4f}")

    s = 0.5
    print(f"At s=0.5, V_pre=-65 mV (no spike):")
    print(f"  ds/dt = {syn.ds_dt(s, V_rest):.4f}")

    # Test synaptic current
    s = 0.8
    V_post = -65.0
    I = syn.current(s, V_post)
    print(f"\nSynaptic current (s=0.8, V_post=-65 mV):")
    print(f"  I_syn = {I:.4f} pA (should be positive = depolarizing)")

    # Test inhibitory
    syn_inh = inhibitory_synapse(g_max=1.0)
    I_inh = syn_inh.current(0.8, -65.0)
    print(f"\nInhibitory current (s=0.8, V_post=-65 mV):")
    print(f"  I_syn = {I_inh:.4f} pA (should be negative = hyperpolarizing)")

    print("\nSynapse OK!")
