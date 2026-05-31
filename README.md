# Bilinear DMDc on FitzHugh-Nagumo Neural Dynamics

A high-performance JAX-based simulation, parameter estimation, and **Bilinear Dynamic Mode Decomposition with Control (Bilinear DMDc)** analysis framework for the FitzHugh-Nagumo (FHN) excitability model.

This repository implements adaptive ODE solvers, gradient-based parameter estimation, and advanced delay-embedded Koopman operator mathematics to capture non-linear state-control interactions in the FHN neural system.

---

## 1. Mathematical Framework

### FitzHugh-Nagumo Dynamics
The neural excitability model is governed by:

$$\frac{dv}{dt} = v - \frac{v^3}{3} - w + I_{ext}(t)$$
$$\frac{dw}{dt} = \frac{v + a - b \cdot w}{\tau}$$

where:
- $v(t)$ is the membrane potential.
- $w(t)$ is the recovery variable.
- $I_{ext}(t)$ is a time-varying stimulation current.

### Bilinear DMDc
Standard DMDc maps linear control inputs. To model non-linear modulation between state activation modes and control currents, we implement **Bilinear DMDc**:
$$x_{t+1} \approx A x_t + B u_t + C (x_t \otimes u_t)$$
where:
- $x_t \in \mathbb{R}^H$ is the delay-embedded Hankel state vector of potential $v$.
- $u_t \in \mathbb{R}^1$ is the scalar stimulation current $I_{ext}(t)$.
- $x_t \otimes u_t = u_t x_t \in \mathbb{R}^H$ represents the bilinear state-input interaction vector.

By constructing the augmented state-control matrix $\Omega$:

$$\Omega = \begin{bmatrix} X \\ U_c \\ X \otimes U_c \end{bmatrix}$$

we compute SVD-truncated, low-dimensional operators:
- $\tilde{A} \in \mathbb{R}^{r \times r}$ (intrinsic autonomous dynamics)
- $\tilde{B} \in \mathbb{R}^{r \times 1}$ (linear control coupling)
- $\tilde{C} \in \mathbb{R}^{r \times r}$ (bilinear interaction matrix representing cross-mode sensitivities)

---

## 2. Repository Structure

```
├── dynamics.py          # ODE vector fields, stimulus currents, and Bilinear DMDc math
├── simulation.py        # Diffrax adaptive ODE integration & Optax parameter estimation
├── plotting.py          # 4-panel visual dashboard, phase nullclines, and DMDc heatmaps
├── fitzhugh_nagumo.py   # CLI entrypoint and main orchestrator
├── requirements.txt     # Python dependencies
├── data/                # Generated CSV time-series and transition matrices
└── plots/               # Saved PNG scientific plots and spectra figures
```

---

## 3. Getting Started

### Installation
Ensure you have Python 3.9+ and the needed scientific packages installed:
```bash
pip install -r requirements.txt
```

### Run Simulation & Bilinear DMDc
Execute the main entrypoint with a dynamic sine stimulation current, delay embedding dimension $H=30$, and SVD ranks $r=6$ and $p=10$:
```bash
python fitzhugh_nagumo.py --I-type sine --dmdc --dmd-H 30 --dmd-r 6 --dmd-p 10
```

### Options
- `--I-type`: Type of stimulation current (`constant`, `step`, `sine`, `pulse`).
- `--fit-demo`: Run gradient-descent parameters fitting demo matching a noisy trajectory.
- `--no-plot`: Run headlessly and skip Matplotlib rendering.
- `--output`: Customize the path to save data (default: `data/fhn_data.csv`).
- `--save-plot`: Customize the path to save plots (default: `plots/fhn_plot.png`).
