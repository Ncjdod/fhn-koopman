import numpy as np
import matplotlib.pyplot as plt
from qho_operators import QuantumHarmonicOscillator

def main():
    m = 9.10938356e-31
    omega = 1.0e15
    hbar = 1.054571817e-34
    energy_scale = hbar * omega
    length_scale = np.sqrt(hbar / (m * omega))

    n_values = list(range(1, 121, 1))

    # 1. Ground State Energy Convergence vs N for different lambda (Log Scale)
    plt.figure(figsize=(10, 6))
    lambda_values = [5.0, 8.0, 10.0, 100.0, 1000.0]
    for lambda_val in lambda_values:
        quartic_scale = energy_scale / (length_scale ** 4)
        alpha_physical = lambda_val * quartic_scale
        energies = []
        for N in n_values:
            qho = QuantumHarmonicOscillator(N=N, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
            val, _ = qho.get_perturbed_state_block(0)
            energies.append(val / 1.602176634e-19)
        plt.plot(n_values, energies, "-", label=f"lambda = {lambda_val}")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Ground State Energy (eV) [Log Scale]")
    plt.title("Ground State Energy Convergence vs N")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_lambda.png", dpi=300)
    plt.close()

    # 2. Excited States Convergence vs N for lambda = 5.0
    plt.figure(figsize=(10, 6))
    quartic_scale = energy_scale / (length_scale ** 4)
    alpha_physical = 5.0 * quartic_scale
    for k in [0, 1, 2, 3, 4]:
        energies = []
        n_values_state = [N for N in n_values if N > k]
        for N in n_values_state:
            qho = QuantumHarmonicOscillator(N=N, alpha=alpha_physical, m=m, omega=omega, hbar=hbar)
            val, _ = qho.get_perturbed_state_block(k)
            energies.append(val / 1.602176634e-19)
        plt.plot(n_values_state, energies, "-", label=f"State k={k}")
    plt.xlabel("N")
    plt.ylabel("Energy (eV)")
    plt.title("Excited States Energy Convergence vs N (lambda = 5.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_states.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
