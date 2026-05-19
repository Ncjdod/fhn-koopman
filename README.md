# Projects

A collection of scientific computing, biophysical modeling, and physics simulation tools.

## General Themes

The projects in this repository focus on two primary computational areas:

### 1. Neural Networks and Ordinary Differential Equations

Located in `neural-networks/`.

This section explores the integration of neural networks with physical dynamics, specifically continuous-depth models trained on neural biophysics data.
* **FitzHugh-Nagumo Model (`neural-ode/fbh-model/`)**: A simplified model of membrane excitability used to explore basic Neural ODE configurations.
* **Hodgkin-Huxley Model (`neural-ode/hh-model/` & `neural-ode/hh-field-model/`)**: Trajectory-fitting models that reconstruct 4D neural dynamics and fine-tune parameters using real electrophysiology recordings.

### 2. Scientific and Biophysical Simulations

Located in `scientific-simulations/`.

Classical physics and statistical mechanics simulations implemented with high-performance numerical routines.
* **Ising Model (`simulations/ising_simulation.py`)**: A 2D spin lattice simulation using Monte Carlo Metropolis dynamics optimized with JAX.
* **Reaction-Diffusion System (`simulations/gray_scott_model.py`)**: An implementation of the Gray-Scott model showing spatial pattern formation.
* **Chaos and PDEs (`simulations/`)**: Simulators for chaotic attractors (Lorenz and Hénon-Heiles systems) and numerical solutions to the 1D heat equation.

## Installation and Setup

To install the scientific stack and run the projects, install the required packages:

```bash
pip install numpy scipy pandas matplotlib jax optax diffrax equinox h5py
```
