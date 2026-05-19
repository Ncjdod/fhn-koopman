import numpy as np
from numpy import linalg

class QuantumHarmonicOscillator:
    def __init__(self, N: int, alpha: float = 1.0):
        if N < 1:
            raise ValueError("Dimension N must be at least 1.")
        self.N = N
        self.alpha = alpha

    @property
    def a(self) -> np.ndarray:
        superdiagonal_elements = np.sqrt(np.arange(1, self.N))
        return np.diag(superdiagonal_elements, k=1)

    @property
    def a_dagger(self) -> np.ndarray:
        return self.a.conj().T

    @property
    def x(self) -> np.ndarray:
        scaling_factor = 1.0 / np.sqrt(2.0)
        return scaling_factor * (self.a + self.a_dagger)

    @property
    def x4(self) -> np.ndarray:
        return linalg.matrix_power(self.x, 4)

    @property
    def h0(self) -> np.ndarray:
        diagonal_elements = 0.5 + np.arange(self.N)
        return np.diag(diagonal_elements)

    @property
    def h(self) -> np.ndarray:
        return self.h0 + self.alpha * self.x4

    @property
    def eigenvalues(self) -> np.ndarray:
        return linalg.eigh(self.h)[0]
