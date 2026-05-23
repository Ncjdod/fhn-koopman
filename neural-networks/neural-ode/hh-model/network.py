"""
Multicellular Neuron Network

Composes trained single-neuron models into coupled networks with
synaptic connections. The full network is a single ODE system
solved jointly with Diffrax.

State layout: [neuron_0(4), neuron_1(4), ..., syn_0(1), syn_1(1), ...]
Total state dimension: 4*N + M  (N neurons, M synapses)

Usage:
    from network import NeuronNetwork, build_network, integrate_network

    # Build a 2-neuron network
    net = build_network(
        neuron_model=trained_model,
        n_neurons=2,
        connections=[(0, 1, 'excitatory')],  # neuron 0 -> neuron 1
    )

    # Integrate
    ys = integrate_network(net, y0, t_span, I_ext_fn_per_neuron)
"""

import jax
import jax.numpy as jnp
import diffrax

from neuron import SingleNeuron
from synapse import Synapse, excitatory_synapse, inhibitory_synapse
from HH_NeuralODE import HHNeuralODE


class NeuronNetwork:
    """
    Network of coupled neurons with synaptic connections.

    The network vector field:
      1. Extracts per-neuron states and synaptic states
      2. Computes synaptic currents from presynaptic voltages
      3. Sums external + synaptic currents per neuron
      4. Evaluates each neuron's dynamics
      5. Evaluates each synapse's dynamics
      6. Concatenates into full state derivative
    """

    def __init__(self, neurons, synapses, connectivity):
        """
        Args:
            neurons:      List of SingleNeuron instances
            synapses:     List of Synapse instances
            connectivity: List of (pre_idx, post_idx, syn_idx) tuples
                          pre_idx:  index of presynaptic neuron
                          post_idx: index of postsynaptic neuron
                          syn_idx:  index into synapses list
        """
        self.neurons = neurons
        self.synapses = synapses
        self.connectivity = connectivity
        self.n_neurons = len(neurons)
        self.n_synapses = len(synapses)
        self.state_dim = 4 * self.n_neurons + self.n_synapses

    def __call__(self, t, state, I_ext_per_neuron):
        """
        Compute full network state derivative.

        Args:
            t:                  Current time (scalar, ms)
            state:              Full network state (shape: (4*N + M,))
            I_ext_per_neuron:   External current for each neuron (shape: (N,), pA)

        Returns:
            d_state/dt: Full derivative (shape: (4*N + M,))
        """
        N = self.n_neurons
        M = self.n_synapses

        # --- Extract per-neuron and synaptic states ---
        neuron_states = state[:4 * N].reshape(N, 4)  # (N, 4)
        syn_states = state[4 * N:]                     # (M,)

        # --- Compute synaptic currents for each neuron ---
        I_syn = jnp.zeros(N)
        for pre_idx, post_idx, syn_idx in self.connectivity:
            V_post = neuron_states[post_idx, 0]
            s = syn_states[syn_idx]
            I_syn = I_syn.at[post_idx].add(
                self.synapses[syn_idx].current(s, V_post)
            )

        # --- Compute neuron dynamics ---
        d_neurons = []
        for i in range(N):
            I_total = I_ext_per_neuron[i] + I_syn[i]
            d_neuron_i = self.neurons[i](t, neuron_states[i], I_total)
            d_neurons.append(d_neuron_i)
        d_neuron_flat = jnp.concatenate(d_neurons)  # (4*N,)

        # --- Compute synapse dynamics ---
        d_syns = []
        for pre_idx, post_idx, syn_idx in self.connectivity:
            V_pre = neuron_states[pre_idx, 0]
            s = syn_states[syn_idx]
            d_s = self.synapses[syn_idx].ds_dt(s, V_pre)
            d_syns.append(d_s)

        if d_syns:
            d_syn_flat = jnp.array(d_syns)  # (M,)
        else:
            d_syn_flat = jnp.array([])

        return jnp.concatenate([d_neuron_flat, d_syn_flat])

    def initial_state(self, V0_per_neuron=None):
        """
        Generate initial state for the full network.

        Args:
            V0_per_neuron: Resting potential per neuron (shape: (N,))
                           Default: -65.0 mV for all

        Returns:
            y0: Full initial state (shape: (4*N + M,))
        """
        if V0_per_neuron is None:
            V0_per_neuron = [-65.0] * self.n_neurons

        neuron_states = []
        for i in range(self.n_neurons):
            y0_i = self.neurons[i].initial_state(V0_per_neuron[i])
            neuron_states.append(y0_i)

        syn_states = jnp.zeros(self.n_synapses)

        return jnp.concatenate(neuron_states + [syn_states])

    def extract_voltages(self, trajectory):
        """
        Extract voltage traces from a full network trajectory.

        Args:
            trajectory: Output from integrate_network, shape (T, 4*N+M)

        Returns:
            voltages: shape (T, N) — voltage of each neuron over time
        """
        N = self.n_neurons
        neuron_states = trajectory[:, :4 * N].reshape(-1, N, 4)
        return neuron_states[:, :, 0]  # Voltage is index 0

    def extract_synaptic_states(self, trajectory):
        """
        Extract synaptic gating variables from trajectory.

        Args:
            trajectory: shape (T, 4*N+M)

        Returns:
            syn_states: shape (T, M)
        """
        return trajectory[:, 4 * self.n_neurons:]


def build_network(neuron_model, n_neurons, connections, g_exc=0.5, g_inh=1.0):
    """
    Build a neuron network from a trained single-neuron model.

    All neurons share the same learned dynamics (weight sharing).

    Args:
        neuron_model: Trained HHNeuralODE instance
        n_neurons:    Number of neurons in the network
        connections:  List of (pre, post, type) tuples
                      type: 'excitatory' or 'inhibitory'
        g_exc:        Excitatory synaptic conductance (pA/mV)
        g_inh:        Inhibitory synaptic conductance (pA/mV)

    Returns:
        NeuronNetwork instance
    """
    # Create neurons (all sharing the same dynamics)
    neurons = [
        SingleNeuron(dynamics=neuron_model, neuron_id=i)
        for i in range(n_neurons)
    ]

    # Create synapses from connection spec
    synapses = []
    connectivity = []
    for idx, (pre, post, syn_type) in enumerate(connections):
        if syn_type == 'excitatory':
            syn = excitatory_synapse(g_max=g_exc)
        elif syn_type == 'inhibitory':
            syn = inhibitory_synapse(g_max=g_inh)
        else:
            raise ValueError(f"Unknown synapse type: {syn_type}")
        synapses.append(syn)
        connectivity.append((pre, post, idx))

    return NeuronNetwork(neurons, synapses, connectivity)


def integrate_network(network, y0, t_span, I_ext_fn_per_neuron,
                      dt0=0.01, rtol=1e-3, atol=1e-5):
    """
    Integrate the full network ODE system.

    Args:
        network:              NeuronNetwork instance
        y0:                   Initial state (shape: (4*N+M,))
        t_span:               Output time points (shape: (T,))
        I_ext_fn_per_neuron:  Function t -> (N,) external currents per neuron
        dt0:                  Initial step size
        rtol, atol:           Integration tolerances

    Returns:
        ys: Full trajectory (shape: (T, 4*N+M))
    """
    def vector_field(t, y, args):
        I_ext = I_ext_fn_per_neuron(t)
        return network(t, y, I_ext)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_span)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=65536,
        throw=False,
    )

    return sol.ys


# ============================================================
# Demo: 2-Neuron Network
# ============================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from HH_NeuralODE import create_model

    print("Neuron Network - 2-Neuron Demo")
    print("=" * 50)

    # Create untrained model (for structure test)
    key = jax.random.PRNGKey(42)
    model = create_model(key=key)

    # Build 2-neuron network: Neuron 0 -> Neuron 1 (excitatory)
    net = build_network(
        neuron_model=model,
        n_neurons=2,
        connections=[(0, 1, 'excitatory')],
        g_exc=0.5,
    )

    print(f"Network: {net.n_neurons} neurons, {net.n_synapses} synapses")
    print(f"State dim: {net.state_dim} (4*{net.n_neurons} + {net.n_synapses})")

    # Initial state
    y0 = net.initial_state()
    print(f"Initial state shape: {y0.shape}")
    print(f"  Neuron 0: V={y0[0]:.1f}, m={y0[1]:.4f}, h={y0[2]:.4f}, n={y0[3]:.4f}")
    print(f"  Neuron 1: V={y0[4]:.1f}, m={y0[5]:.4f}, h={y0[6]:.4f}, n={y0[7]:.4f}")
    print(f"  Synapse s: {y0[8]:.4f}")

    # External current: only neuron 0 receives stimulus
    def I_ext_fn(t):
        # Step current to neuron 0 starting at t=5ms
        I0 = jnp.where(t > 5.0, 200.0, 0.0)  # pA
        I1 = 0.0  # No external input to neuron 1
        return jnp.array([I0, I1])

    # Integrate
    t_span = jnp.linspace(0.0, 50.0, 500)  # 50ms
    print(f"\nIntegrating 50ms...")
    ys = integrate_network(net, y0, t_span, I_ext_fn)
    print(f"Output shape: {ys.shape}")

    # Extract voltages
    voltages = net.extract_voltages(ys)
    syn_states = net.extract_synaptic_states(ys)

    print(f"Neuron 0 V range: [{voltages[:, 0].min():.1f}, {voltages[:, 0].max():.1f}] mV")
    print(f"Neuron 1 V range: [{voltages[:, 1].min():.1f}, {voltages[:, 1].max():.1f}] mV")
    print(f"Synapse s range:  [{syn_states[:, 0].min():.4f}, {syn_states[:, 0].max():.4f}]")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_span, voltages[:, 0], 'b-', lw=1.5, label='Neuron 0 (stimulated)')
    axes[0].plot(t_span, voltages[:, 1], 'r-', lw=1.5, label='Neuron 1 (synapse only)')
    axes[0].set_ylabel('V (mV)')
    axes[0].set_title('2-Neuron Network: Excitatory Coupling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_span, syn_states[:, 0], 'g-', lw=1.5)
    axes[1].set_ylabel('Synaptic s')
    axes[1].set_title('Synaptic Gating Variable')
    axes[1].grid(True, alpha=0.3)

    I_ext_trace = jax.vmap(lambda t: I_ext_fn(t)[0])(t_span)
    axes[2].plot(t_span, I_ext_trace, 'k-', lw=1.5)
    axes[2].set_ylabel('I_ext (pA)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('External Stimulus (Neuron 0 only)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('HH_model/network_demo.png', dpi=150)
    plt.close()
    print("\nSaved network_demo.png")
    print("Network OK!")
