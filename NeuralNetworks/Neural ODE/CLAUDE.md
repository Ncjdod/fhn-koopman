# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural ODE implementations for learning Hodgkin-Huxley (HH) neuron dynamics from electrophysiology data. The project uses JAX/Equinox/Optax (pure JAX stack) to train neural networks that approximate the HH vector field, then fits them to real Allen Brain Institute recordings.

## Tech Stack

- **JAX** + **Equinox** (neural network modules) + **Optax** (optimizers) + **Diffrax** (ODE solvers)
- Python 3.12
- Allen Brain Observatory NWB data (electrophysiology recordings)
- No pip requirements file — install: `pip install jax[cuda12] equinox optax diffrax h5py scipy matplotlib`

## Repository Structure

Three model directories, representing evolutionary stages:

### `FBH_model/` — FitzHugh-Nagumo (starter/prototype)
- Simple 2D Neural ODE on the FitzHugh-Nagumo system (toy model)
- Uses Keras with JAX backend. `NeuralODE.py` = basic, `PINN_NeuralODE.py` = physics-informed variant

### `HH_model/` — Hodgkin-Huxley Neural ODE (trajectory fitting)
- Learns 4D HH dynamics [V, m, h, n] by fitting to Allen Brain voltage traces
- **Architecture**: `HH_NeuralODE.py` — `HHNeuralODE` with Fourier features + eqx.nn.MLP + Diffrax integration
- **Training** (`train.py`): Multiple shooting with adversarial physics loss (Self-Adaptive PINN), curriculum learning
- **Key modules**: `multiple_shooting.py` (custom Heun integrator via `jax.lax.scan`), `physics_loss.py` (minimax training step), `curriculum.py` (progressive time window + segment ramp)
- **Network composition** (`network.py`, `neuron.py`, `synapse.py`): Composes trained single neurons into coupled networks with synaptic connections

### `HH_Field_Model/` — Vector Field Learning (current approach)
- Two-phase training that first distills the HH vector field, then fine-tunes on real data
- **Architecture**: `model.py` — `VectorFieldNet` with explicit stacked weights + `jax.lax.scan` (avoids GPU compilation hang from MLP unrolling)
- **Phase 1** (`train_field.py`): Supervised regression on (state, derivative) pairs sampled from HH equations. Variance-normalized MSE loss auto-balances V vs gating scales
- **Phase 2** (`train_boundary.py`): Fine-tune on Allen Brain data with latent gating variables, unit conversion learning (pA ↔ uA/cm²), and anti-forgetting field loss
- **Config**: `config.py` — `Phase1Config` and `Phase2Config` classes with all hyperparameters

## Running Training

```bash
# HH_Field_Model (current approach)
cd HH_Field_Model
python train_field.py              # Phase 1: vector field distillation
python train_boundary.py           # Phase 2: Allen Brain fine-tuning (loads Phase 1 checkpoint)
python train_boundary.py --from_scratch  # Both phases sequentially

# HH_model (trajectory fitting approach)
cd HH_model
python train.py

# Test individual modules
python model.py          # Architecture test
python hh_reference.py   # HH equations test
python multiple_shooting.py  # Integration test
```

## Key Architectural Decisions

- **lax.scan over hidden layers**: `VectorFieldNet` stacks weight matrices into arrays and iterates via `jax.lax.scan` instead of using `eqx.nn.MLP`. This produces a compact XLA graph that compiles on GPU; the MLP approach causes graph explosion and compilation hangs.
- **Native batch ops over vmap**: `predict_batch()` uses matrix multiplication that naturally broadcasts over batch dimension. No `jax.vmap` needed — avoids GPU compilation issues.
- **Gate clipping at integration time, not in forward pass**: Gating variables [m, h, n] are clipped to [0,1] after each Euler/Heun step during evaluation, but NOT in the network's forward pass. This prevents vanishing gradients during training.
- **Derivative clipping in forward pass**: Output derivatives are hard-clipped (dV: ±500, gates: ±25) to prevent integration divergence.
- **Stop-gradient on physics loss trajectory**: In multiple shooting, physics residual gradients flow only through the direct `model(t,y,I)` call, not through the integration chain. Prevents gradient explosion from chaining Jacobians through all integration steps.
- **Fourier features**: Random Fourier feature encoding on inputs captures sharp HH nonlinearities. The frequency matrix `B` is fixed (non-trainable).

## Unit System

The HH model uses mV/ms/uA·cm⁻²/mS·cm⁻². Allen Brain data uses pA (absolute current). The conversion between pA and uA/cm² depends on membrane area, which is **learned** during training (initialized at ~2000 μm² typical cortical soma). See `PhysicsParams` in `physics_loss.py` and `ConversionFactor` in `latent_state.py`.

## Checkpoints

Serialized via `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves` to `.eqx` files. To load, create a skeleton model with matching architecture, then deserialize into it.
