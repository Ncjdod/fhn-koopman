import numpy as np
from qho_operators import QuantumHarmonicOscillator

def run_demo():
    N = 6
    alpha = 1.5
    qho = QuantumHarmonicOscillator(N=N, alpha=alpha)
    
    print("QHO:", qho.N, qho.alpha)
    print("a:")
    print(np.round(qho.a, 4))
    print("a_dagger:")
    print(np.round(qho.a_dagger, 4))
    print("[a, a_dagger]:")
    print(np.round(qho.a @ qho.a_dagger - qho.a_dagger @ qho.a, 4))
    print("x:")
    print(np.round(qho.x, 4))
    print("x4:")
    print(np.round(qho.x4, 4))

if __name__ == "__main__":
    run_demo()
