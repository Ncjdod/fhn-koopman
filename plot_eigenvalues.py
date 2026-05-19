import numpy as np
import matplotlib.pyplot as plt
from qho_operators import QuantumHarmonicOscillator

def main():
    n_values = list(range(10, 101, 5))
    alpha_values = [0.1, 0.5, 1.0]

    plt.figure(figsize=(10, 6))

    for alpha in alpha_values:
        min_eigenvalues = []
        for N in n_values:
            qho = QuantumHarmonicOscillator(N=N, alpha=alpha)
            min_eigenvalues.append(qho.eigenvalues[0])
        plt.plot(n_values, min_eigenvalues, "o-", label=f"alpha = {alpha}")

    plt.xlabel("N")
    plt.ylabel("Minimal Eigenvalue (Ground State Energy)")
    plt.title("Minimal Eigenvalue of Perturbed QHO vs Dimension N")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimal_eigenvalues.png", dpi=300)

if __name__ == "__main__":
    main()
