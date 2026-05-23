import numpy as np
import matplotlib.pyplot as plt
from qho_operators import QuantumHarmonicOscillator

def main():
    m = 9.10938356e-31
    omega = 1.0e15
    hbar = 1.054571817e-34

    energy_scale = hbar * omega
    length_scale = np.sqrt(hbar / (m * omega))
    quartic_scale = energy_scale / (length_scale ** 4)

    n_values = list(range(1, 101, 1))
    alpha_coefficients = [5.0, 8.0, 10.0]

    plt.figure(figsize=(10, 6))

    for alpha_coeff in alpha_coefficients:
        alpha_physical = alpha_coeff * quartic_scale
        min_eigenvalues = []
        for N in n_values:
            qho = QuantumHarmonicOscillator(N=N, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
            min_eigenvalues_ev = qho.eigenvalues[0] / 1.602176634e-19
            min_eigenvalues.append(min_eigenvalues_ev)
        plt.plot(n_values, min_eigenvalues, "-", label=f"alpha_coeff = {alpha_coeff} (alpha = {alpha_physical:.3e} J/m^4)")
        print(f"alpha_coeff = {alpha_coeff} | Converged ground state energy (N=100) = {min_eigenvalues[-1]:.10f} eV")

    plt.xlabel("N")
    plt.ylabel("Minimal Eigenvalue (Ground State Energy in eV)")
    plt.title("Minimal Eigenvalue of Perturbed QHO vs Dimension N (N from 1 to 100)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimal_eigenvalues.png", dpi=300)

if __name__ == "__main__":
    main()
