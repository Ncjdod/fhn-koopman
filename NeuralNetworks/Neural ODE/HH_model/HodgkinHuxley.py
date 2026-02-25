"""
Hodgkin-Huxley Model

Callable class implementing the standard HH equations for use as a
physics prior in the Neural ODE loss function.

All units: mV (voltage), ms (time), uA/cm^2 (current density),
           mS/cm^2 (conductance)
"""

import jax.numpy as jnp


class HodgkinHuxley:
    """
    Standard Hodgkin-Huxley neuron model.
    
    Usage:
        hh = HodgkinHuxley()
        dydt = hh(t, y, I_ext)    # y = [V, m, h, n]
        
        # Or access individual components:
        alpha_m, beta_m = hh.alpha_m(V), hh.beta_m(V)
        I_Na = hh.I_Na(V, m, h)
    """
    
    # ---- Default Parameters ----
    # Capacitance (uF/cm^2)
    C_m = 1.0
    
    # Maximum conductances (mS/cm^2)
    g_Na = 120.0
    g_K  = 36.0
    g_L  = 0.3
    
    # Reversal potentials (mV)
    E_Na = 50.0
    E_K  = -77.0
    E_L  = -54.4
    
    # ---- Rate Functions (alpha, beta) ----
    
    @staticmethod
    def alpha_m(V):
        """Na+ activation rate."""
        dV = V + 40.0
        safe_dV = jnp.where(jnp.abs(dV) < 1e-6, 1.0, dV)
        return jnp.where(
            jnp.abs(dV) < 1e-6,
            1.0,  
            0.1 * safe_dV / (1.0 - jnp.exp(-safe_dV / 10.0))
        )
    
    @staticmethod
    def beta_m(V):
        """Na+ activation decay rate."""
        return 4.0 * jnp.exp(-(V + 65.0) / 18.0)
    
    @staticmethod
    def alpha_h(V):
        """Na+ inactivation rate."""
        return 0.07 * jnp.exp(-(V + 65.0) / 20.0)
    
    @staticmethod
    def beta_h(V):
        """Na+ inactivation decay rate."""
        return 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
    
    @staticmethod
    def alpha_n(V):
        """K+ activation rate."""
        dV = V + 55.0
        safe_dV = jnp.where(jnp.abs(dV) < 1e-6, 1.0, dV)
        return jnp.where(
            jnp.abs(dV) < 1e-6,
            0.1,  
            0.01 * safe_dV / (1.0 - jnp.exp(-safe_dV / 10.0))
        )
    
    @staticmethod
    def beta_n(V):
        """K+ activation decay rate."""
        return 0.125 * jnp.exp(-(V + 65.0) / 80.0)
    
    # ---- Steady-State & Time Constants ----
    
    @staticmethod
    def m_inf(V):
        a = HodgkinHuxley.alpha_m(V)
        return a / (a + HodgkinHuxley.beta_m(V))
    
    @staticmethod
    def h_inf(V):
        a = HodgkinHuxley.alpha_h(V)
        return a / (a + HodgkinHuxley.beta_h(V))
    
    @staticmethod
    def n_inf(V):
        a = HodgkinHuxley.alpha_n(V)
        return a / (a + HodgkinHuxley.beta_n(V))
    
    # ---- Ionic Currents ----
    
    def I_Na(self, V, m, h):
        """Sodium current."""
        return self.g_Na * (m ** 3) * h * (V - self.E_Na)
    
    def I_K(self, V, n):
        """Potassium current."""
        return self.g_K * (n ** 4) * (V - self.E_K)
    
    def I_L(self, V):
        """Leak current."""
        return self.g_L * (V - self.E_L)
    
    # ---- Full Dynamics ----
    
    def __call__(self, t, y, I_ext=0.0):
        """
        Compute dy/dt for the full HH system.
        
        Args:
            t:     Time (ms) - unused, system is autonomous
            y:     State vector [V, m, h, n]
            I_ext: External current (uA/cm^2)
        
        Returns:
            dydt: [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        V, m, h, n = y[0], y[1], y[2], y[3]
        
        # Membrane voltage
        dVdt = (I_ext - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        
        # Gating variables
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        
        return jnp.array([dVdt, dmdt, dhdt, dndt])
    
    def dVdt(self, V, m, h, n, I_ext=0.0):
        """
        Compute only dV/dt (useful for physics loss on voltage only).
        
        Args:
            V, m, h, n: State variables
            I_ext:      External current (uA/cm^2)
        
        Returns:
            dV/dt (scalar)
        """
        return (I_ext - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
    
    def resting_state(self, V_rest=-65.0):
        """
        Return the steady-state gating variables at a given resting voltage.
        
        Args:
            V_rest: Resting membrane potential (mV)
        
        Returns:
            y0: [V_rest, m_inf, h_inf, n_inf]
        """
        return jnp.array([
            V_rest,
            self.m_inf(V_rest),
            self.h_inf(V_rest),
            self.n_inf(V_rest),
        ])


if __name__ == "__main__":
    print("Hodgkin-Huxley Model Test")
    print("=" * 40)
    
    hh = HodgkinHuxley()
    
    y0 = hh.resting_state()
    print(f"Resting state: V={y0[0]:.1f}mV, m={y0[1]:.4f}, h={y0[2]:.4f}, n={y0[3]:.4f}")
    
    # Dynamics at rest (should be ~0)
    dydt = hh(0.0, y0, I_ext=0.0)
    print(f"dy/dt at rest: {dydt}")
    
    # Dynamics with current injection
    dydt_stim = hh(0.0, y0, I_ext=10.0)
    print(f"dy/dt with I_ext=10: dV/dt={dydt_stim[0]:.2f} mV/ms")
    
    # Individual currents
    V, m, h, n = y0
    print(f"\nIonic currents at rest:")
    print(f"  I_Na = {hh.I_Na(V, m, h):.4f} uA/cm^2")
    print(f"  I_K  = {hh.I_K(V, n):.4f} uA/cm^2")
    print(f"  I_L  = {hh.I_L(V):.4f} uA/cm^2")
    
    print("\nHH Model OK!")
