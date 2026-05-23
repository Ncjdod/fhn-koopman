import numpy as np
from numpy import linalg

class QuantumHarmonicOscillator:
    def __init__(self, N: int, alpha: float = 1.0, k: int = 4, m: float = 9.10938356e-31, omega: float = 1.0e15, hbar: float = 1.054571817e-34):
        if N < 1:
            raise ValueError("Dimension N must be at least 1.")
        self.N = N
        self.alpha = alpha
        self.k = k
        self.m = m
        self.omega = omega
        self.hbar = hbar

    @property
    def a(self) -> np.ndarray:
        superdiagonal_elements = np.sqrt(np.arange(1, self.N))
        return np.diag(superdiagonal_elements, k=1)

    @property
    def a_dagger(self) -> np.ndarray:
        return self.a.conj().T

    @property
    def x(self) -> np.ndarray:
        scaling_factor = np.sqrt(self.hbar / (2.0 * self.m * self.omega))
        return scaling_factor * (self.a + self.a_dagger)

    @property
    def xk(self) -> np.ndarray:
        large_N = self.N + self.k
        superdiagonal_elements = np.sqrt(np.arange(1, large_N))
        a_large = np.diag(superdiagonal_elements, k=1)
        a_dagger_large = a_large.conj().T
        scaling_factor = np.sqrt(self.hbar / (2.0 * self.m * self.omega))
        x_large = scaling_factor * (a_large + a_dagger_large)
        xk_large = linalg.matrix_power(x_large, self.k)
        return xk_large[:self.N, :self.N]

    @property
    def h0(self) -> np.ndarray:
        diagonal_elements = self.hbar * self.omega * (0.5 + np.arange(self.N))
        return np.diag(diagonal_elements)

    @property
    def h(self) -> np.ndarray:
        return self.h0 + self.alpha * self.xk

    @property
    def eigenvalues(self) -> np.ndarray:
        return linalg.eigh(self.h)[0]

    def get_energy_state(self, index: int) -> tuple[float, np.ndarray]:
        eigenvalues, eigenvectors = linalg.eigh(self.h)
        return eigenvalues[index], eigenvectors[:, index]
