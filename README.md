# Projects

A collection of scientific computing, biophysical modeling, and physics simulation tools.

## General Themes

The projects in this repository focus on two primary computational areas:

### 1. Neural Networks and Ordinary Differential Equations

Located in `neural-networks/`.

This section explores the integration of neural networks with physical dynamics, specifically continuous-depth models trained on neural biophysics data.
* **FitzHugh-Nagumo Model (`neural-ode/fbh-model/`)**: A simplified model of membrane excitability used to explore basic Neural ODE configurations.
* **Hodgkin-Huxley Model (`neural-ode/hh-model/` & `neural-ode/hh-field-model/`)**: Trajectory-fitting models that reconstruct 4D neural dynamics and fine-tune parameters using real electrophysiology recordings.

### 2. Scientific Analysis and Simulations

Located in `scientific-analysis-and-simulations/`.

Classical physics, statistical mechanics, and quantum mechanics simulations implemented with high-performance numerical routines.

#### Basic Simulations (`basic-simulations/`)

Organized into individual subfolders containing the model scripts and their corresponding visualizations:
* **Ising Model (`ising/`)**: A 2D spin lattice simulation using Monte Carlo Metropolis dynamics optimized with JAX, including generated simulation animations.
* **Reaction-Diffusion System (`gray-scott/`)**: An implementation of the Gray-Scott model showing spatial pattern formation, including live pattern renders.
* **Heated Rod PDE (`heated-rod-1d/`)**: Numerical solution to the 1D heat equation showing thermal stabilization.
* **Chaotic Systems (`henon-hailes/` & `lorenz-attractor/`)**: Simulators for chaotic attractors (Lorenz and Hénon-Heiles systems).
* **Hebbian learning (`binary-hebbian/`)**: Synaptic spike learning simulator.

#### System Analysis (`system-analysis/`)

Focuses on the analytical study of quantum and physical systems:
* **Quantum Harmonic Oscillator (`quantum-harmonic-oscillator/`)**: Numerical solvers for a 1D quantum harmonic oscillator with quartic perturbation using Galerkin projection, including eigenvector parity verification, block-diagonalization benchmarks, and energy state convergence analysis.

## Installation and Setup

To install the scientific stack and run the projects, install the required packages:

```bash
pip install numpy scipy pandas matplotlib jax optax diffrax equinox h5py
```
