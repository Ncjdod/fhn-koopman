import time
import numpy as np
from qho_operators import QuantumHarmonicOscillator

def get_convergence_n(alpha_val, k, threshold=1e-8):
    m = 9.10938356e-31
    omega = 1.0e15
    hbar = 1.054571817e-34
    energy_scale = hbar * omega
    length_scale = np.sqrt(hbar / (m * omega))
    quartic_scale = energy_scale / (length_scale ** 4)
    alpha_physical = alpha_val * quartic_scale

    # Compute reference value at N=120
    qho_ref = QuantumHarmonicOscillator(N=120, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
    ref_energy = qho_ref.get_perturbed_state_block(k)[0] / 1.602176634e-19

    for N in range(k + 1, 120):
        qho = QuantumHarmonicOscillator(N=N, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
        energy = qho.get_perturbed_state_block(k)[0] / 1.602176634e-19
        if abs(energy - ref_energy) < threshold:
            return N
    return 120

def main():
    m = 9.10938356e-31
    omega = 1.0e15
    hbar = 1.054571817e-34
    energy_scale = hbar * omega
    length_scale = np.sqrt(hbar / (m * omega))
    quartic_scale = energy_scale / (length_scale ** 4)

    # 1. Parity and eigenvectors verification
    alpha_val = 5.0
    alpha_physical = alpha_val * quartic_scale
    qho_10 = QuantumHarmonicOscillator(N=10, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
    _, v0 = qho_10.get_perturbed_state_block(0)
    _, v1 = qho_10.get_perturbed_state_block(1)

    parity_table_rows = []
    for n in range(10):
        parity_table_rows.append(f"| {n} | {v0[n]:.3e} | {v1[n]:.3e} |")

    # 2. Timing benchmarks
    timing_rows = []
    n_bench_values = [100, 200, 300, 400, 500]
    for N in n_bench_values:
        qho_bench = QuantumHarmonicOscillator(N=N, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
        
        # Benchmark full
        start = time.perf_counter()
        _ = qho_bench.eigenvalues
        t_full = (time.perf_counter() - start) * 1000.0  # ms
        
        # Benchmark block
        start = time.perf_counter()
        _ = qho_bench.eigenvalues_block
        t_block = (time.perf_counter() - start) * 1000.0  # ms
        
        speedup = t_full / t_block
        timing_rows.append(f"| {N} | {t_full:.2f} ms | {t_block:.2f} ms | {speedup:.2f}x |")

    # 3. Convergence for different lambda (alpha_coeff)
    lambda_values = [5.0, 8.0, 10.0, 100.0, 1000.0]
    lambda_conv_rows = []
    for l_val in lambda_values:
        n_conv = get_convergence_n(l_val, k=0)
        lambda_conv_rows.append(f"| {l_val} | {n_conv} |")

    # 4. Convergence for higher states (alpha_coeff = 5.0)
    state_indices = [0, 1, 2, 3, 4]
    state_conv_rows = []
    for k_val in state_indices:
        n_conv = get_convergence_n(5.0, k=k_val)
        state_conv_rows.append(f"| State k={k_val} | {n_conv} |")

    # 5. Write results.md
    with open("results.md", "w") as f:
        f.write("# Galerkin Projection of 1D Quantum Harmonic Oscillator with quartic perturbation\n\n")
        f.write("This report presents numerical results for the 1D quantum harmonic oscillator perturbed by $\\lambda x^4$ using the Galerkin projection in the number state (Fock) basis.\n\n")
        
        f.write("## 1. Eigenvector Parity Verification\n")
        f.write("Due to parity symmetry $[H, \\Pi] = 0$, even states have zero coefficients at odd indices, and odd states have zero coefficients at even indices.\n\n")
        f.write("| Basis Index (n) | State k=0 (Even) | State k=1 (Odd) |\n")
        f.write("|-----------------|------------------|-----------------|\n")
        for row in parity_table_rows:
            f.write(row + "\n")
        f.write("\n")
        
        f.write("## 2. Computational Efficiency of Block Diagonalization\n")
        f.write("Reordering basis into even ($0, 2, 4, \\dots$) and odd ($1, 3, 5, \\dots$) coordinates splits the $N \\times N$ Hamiltonian into two $N/2 \\times N/2$ independent block matrices. Solving them separately yields a theoretical $4\\times$ speedup since eigenvalue computation scales as $O(N^3)$.\n\n")
        f.write("| N | Full Matrix Solver Time | Block Diagonal Solver Time | Speedup Factor |\n")
        f.write("|---|-------------------------|----------------------------|----------------|\n")
        for row in timing_rows:
            f.write(row + "\n")
        f.write("\n")
        
        f.write("## 3. Convergence of Ground State Energy vs Perturbation Strength (\\lambda)\n")
        f.write("The minimum Hilbert space dimension $N$ required to achieve ground state energy convergence to $< 10^{-8}$ eV:\n\n")
        f.write("| Perturbation Strength (\\lambda) | Minimum N for Ground State Convergence |\n")
        f.write("|--------------------------------|----------------------------------------|\n")
        for row in lambda_conv_rows:
            f.write(row + "\n")
        f.write("\n")
        
        f.write("## 4. Convergence of Higher Energy States (\\lambda = 5.0)\n")
        f.write("Minimum $N$ required to achieve state energy convergence to $< 10^{-8}$ eV for higher excited states:\n\n")
        f.write("| State Index (k) | Minimum N for State Convergence |\n")
        f.write("|-----------------|----------------------------------|\n")
        for row in state_conv_rows:
            f.write(row + "\n")

    print("results.md generated successfully.")

if __name__ == "__main__":
    main()
