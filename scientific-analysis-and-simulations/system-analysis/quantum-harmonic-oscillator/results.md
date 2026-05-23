# Galerkin Projection of 1D Quantum Harmonic Oscillator with quartic perturbation

This report presents numerical results for the 1D quantum harmonic oscillator perturbed by $\lambda x^4$ using the Galerkin projection in the number state (Fock) basis.

## 1. Eigenvector Parity Verification
Due to parity symmetry $[H, \Pi] = 0$, even states have zero coefficients at odd indices, and odd states have zero coefficients at even indices.

| Basis Index (n) | State k=0 (Even) | State k=1 (Odd) |
|-----------------|------------------|-----------------|
| 0 | 9.249e-01 | 0.000e+00 |
| 1 | 0.000e+00 | 7.759e-01 |
| 2 | -3.467e-01 | 0.000e+00 |
| 3 | 0.000e+00 | -5.316e-01 |
| 4 | 1.456e-01 | 0.000e+00 |
| 5 | 0.000e+00 | 3.047e-01 |
| 6 | -5.472e-02 | 0.000e+00 |
| 7 | 0.000e+00 | -1.433e-01 |
| 8 | 1.433e-02 | 0.000e+00 |
| 9 | 0.000e+00 | 4.495e-02 |

## 2. Computational Efficiency of Block Diagonalization
Reordering basis into even ($0, 2, 4, \dots$) and odd ($1, 3, 5, \dots$) coordinates splits the $N \times N$ Hamiltonian into two $N/2 \times N/2$ independent block matrices. Solving them separately yields a theoretical $4\times$ speedup since eigenvalue computation scales as $O(N^3)$.

| N | Full Matrix Solver Time | Block Diagonal Solver Time | Speedup Factor |
|---|-------------------------|----------------------------|----------------|
| 100 | 4.15 ms | 4.21 ms | 0.99x |
| 200 | 8.29 ms | 7.86 ms | 1.05x |
| 300 | 20.70 ms | 14.13 ms | 1.46x |
| 400 | 32.48 ms | 21.82 ms | 1.49x |
| 500 | 52.57 ms | 39.72 ms | 1.32x |

## 3. Convergence of Ground State Energy vs Perturbation Strength ($\lambda$)
The minimum Hilbert space dimension $N$ required to achieve ground state energy convergence to $< 10^{-8}$ eV:

| Perturbation Strength (\lambda) | Minimum N for Ground State Convergence |
|--------------------------------|----------------------------------------|
| 5.0 | 59 |
| 8.0 | 77 |
| 10.0 | 83 |
| 100.0 | 119 |
| 1000.0 | 119 |

## 4. Convergence of Higher Energy States ($\lambda = 5.0$)
Minimum $N$ required to achieve state energy convergence to $< 10^{-8}$ eV for higher excited states:

| State Index (k) | Minimum N for State Convergence |
|-----------------|----------------------------------|
| State k=0 | 59 |
| State k=1 | 76 |
| State k=2 | 87 |
| State k=3 | 100 |
| State k=4 | 109 |
