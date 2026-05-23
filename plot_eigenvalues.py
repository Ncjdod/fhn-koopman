import numpy as np
import matplotlib.pyplot as plt
from qho_operators import QuantumHarmonicOscillator

def main():
    m = 9.10938356e-31
    omega = 1.0e15
    hbar = 1.054571817e-34

    energy_scale = hbar * omega
    length_scale = np.sqrt(hbar / (m * omega))

    n_values = list(range(1, 101, 1))
    alpha_coeff = 5.0

    plt.figure(figsize=(10, 6))

    quartic_scale = energy_scale / (length_scale ** 4)
    alpha_physical = alpha_coeff * quartic_scale
    
    for k in [0, 1, 2]:
        energies = []
        n_values_state = [N for N in n_values if N > k]
        for N in n_values_state:
            qho = QuantumHarmonicOscillator(N=N, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
            val, _ = qho.get_perturbed_state_block(k)
            energies.append(val / 1.602176634e-19)
        plt.plot(n_values_state, energies, "-", label=f"State k={k}")
        print(f"State k={k} | Converged energy (N=100) = {energies[-1]:.10f} eV")

    plt.xlabel("N")
    plt.ylabel("Energy in eV")
    plt.title("Perturbed QHO Energy Levels (k = 0, 1, 2) vs Dimension N (alpha_coeff = 5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimal_eigenvalues.png", dpi=300)

if __name__ == "__main__":
    main()
