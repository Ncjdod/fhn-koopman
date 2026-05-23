import numpy as np
from numpy import linalg

class QuantumHarmonicOscillator:
    def __init__(self, N: int, alpha: float = 1.0, m: float = 9.10938356e-31, omega: float = 1.0e15, hbar: float = 1.054571817e-34):
        if N < 1:
            raise ValueError("Dimension N must be at least 1.")
        self.N = N
        self.alpha = alpha
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
    def x4(self) -> np.ndarray:
        large_N = self.N + 4
        superdiagonal_elements = np.sqrt(np.arange(1, large_N))
        a_large = np.diag(superdiagonal_elements, k=1)
        a_dagger_large = a_large.conj().T
        scaling_factor = np.sqrt(self.hbar / (2.0 * self.m * self.omega))
        x_large = scaling_factor * (a_large + a_dagger_large)
        x4_large = linalg.matrix_power(x_large, 4)
        return x4_large[:self.N, :self.N]

    @property
    def h0(self) -> np.ndarray:
        diagonal_elements = self.hbar * self.omega * (0.5 + np.arange(self.N))
        return np.diag(diagonal_elements)

    @property
    def h(self) -> np.ndarray:
        return self.h0 + self.alpha * self.x4

    @property
    def h_even(self) -> np.ndarray:
        even_indices = np.arange(0, self.N, 2)
        return self.h[even_indices, :][:, even_indices]

    @property
    def h_odd(self) -> np.ndarray:
        odd_indices = np.arange(1, self.N, 2)
        return self.h[odd_indices, :][:, odd_indices]

    @property
    def eigenvalues(self) -> np.ndarray:
        return linalg.eigh(self.h)[0]

    @property
    def eigenvalues_block(self) -> np.ndarray:
        even_eigenvalues = linalg.eigh(self.h_even)[0]
        odd_eigenvalues = linalg.eigh(self.h_odd)[0]
        return np.sort(np.concatenate([even_eigenvalues, odd_eigenvalues]))

    def get_perturbed_state(self, k: int) -> tuple[float, np.ndarray]:
        eigenvalues, eigenvectors = linalg.eigh(self.h)
        return eigenvalues[k], eigenvectors[:, k]

    def get_perturbed_state_block(self, k: int) -> tuple[float, np.ndarray]:
        if k % 2 == 0:
            even_index = k // 2
            even_eigenvalues, even_eigenvectors = linalg.eigh(self.h_even)
            state_energy = even_eigenvalues[even_index]
            state_vector = np.zeros(self.N)
            state_vector[np.arange(0, self.N, 2)] = even_eigenvectors[:, even_index]
            return state_energy, state_vector
        else:
            odd_index = k // 2
            odd_eigenvalues, odd_eigenvectors = linalg.eigh(self.h_odd)
            state_energy = odd_eigenvalues[odd_index]
            state_vector = np.zeros(self.N)
            state_vector[np.arange(1, self.N, 2)] = odd_eigenvectors[:, odd_index]
            return state_energy, state_vector
