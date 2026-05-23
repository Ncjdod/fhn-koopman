import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "jax"  # Must be set BEFORE importing keras

import keras
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Ground Truth (FHN) ---
def fhn_dynamics(y, a=0.7, b=0.8, tau=12.5, I_ext=0.5):
    """FitzHugh-Nagumo dynamics. Shape: y is [..., 2]"""
    v = y[..., 0]
    w = y[..., 1]
    dv = v - (v**3)/3 - w + I_ext
    dw = (v + a - b*w) / tau
    return jnp.stack([dv, dw], axis=-1)

# --- 2. Solver (RK4) using JAX ---
def rk4_step(func, y, dt):
    """Single RK4 step. func(y) -> dy/dt"""
    k1 = func(y)
    k2 = func(y + dt*k1/2)
    k3 = func(y + dt*k2/2)
    k4 = func(y + dt*k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def integrate_with_scan(func, y0, dt, n_steps):
    """
    Integrate ODE using lax.scan for XLA optimization.
    Much faster than Python for loop!
    
    Args:
        func: dynamics function y -> dy/dt
        y0: initial condition [batch, 2]
        dt: time step (scalar)
        n_steps: number of integration steps
    
    Returns:
        ys: trajectory [n_steps+1, batch, 2]
    """
    def step_fn(carry, _):
        y = carry
        y_next = rk4_step(func, y, dt)
        return y_next, y_next  # (new_carry, output)
    
    # lax.scan efficiently compiles the loop
    _, ys = lax.scan(step_fn, y0, None, length=n_steps)
    
    # Prepend initial condition
    ys = jnp.concatenate([y0[None, ...], ys], axis=0)
    return ys  # [n_steps+1, batch, 2]

# --- 3. Neural ODE Model ---
@keras.saving.register_keras_serializable()
class NeuralODE(keras.Model):
    def __init__(self, dt, n_steps, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.n_steps = n_steps
        
        # Simple MLP for vector field
        self.net = keras.Sequential([
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(2)
        ])

    def call(self, y0):
        """
        Forward pass: integrate y0 using lax.scan.
        This is XLA-compiled and very fast!
        """
        # Define the dynamics wrapper for our network
        def neural_dynamics(y):
            return self.net(y)
        
        # Use the scan-based integrator
        ys = integrate_with_scan(neural_dynamics, y0, self.dt, self.n_steps)
        
        # Return [batch, time, 2] to match Keras conventions
        return jnp.transpose(ys, (1, 0, 2))

    def get_config(self):
        config = super().get_config()
        config.update({
            "dt": float(self.dt),
            "n_steps": int(self.n_steps),
        })
        return config

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Parameters
    T = 50.0
    steps = 200
    dt = T / (steps - 1)
    t_span = jnp.linspace(0.0, T, steps)
    
    # Generate Ground Truth Data (also using lax.scan for speed!)
    print("Generating ground truth trajectories...")
    
    # Generate 60 random initial conditions across the phase space
    # FHN typical range: v in [-2, 2], w in [-1, 1.5]
    key = jax.random.PRNGKey(42)
    n_samples = 121
    key_v, key_w = jax.random.split(key)
    v0 = jax.random.uniform(key_v, (n_samples,), minval=-2.0, maxval=2.0)
    w0 = jax.random.uniform(key_w, (n_samples,), minval=-1.0, maxval=1.5)
    y0_pre = jnp.stack([v0, w0], axis=-1).astype(jnp.float32)
    y0 = y0_pre[:n_samples -1]
    y0_test = y0_pre[n_samples - 1:n_samples]  # Keep batch dim: shape (1, 2)
    print(f"Generated {n_samples} initial conditions")
    
    # Use our optimized integrator for ground truth
    y_true_transposed = integrate_with_scan(fhn_dynamics, y0, dt, steps - 1)
    y_true = jnp.transpose(y_true_transposed, (1, 0, 2))  # [batch, time, 2]
    print(f"Ground truth shape: {y_true.shape}")

    # Train with Learning Rate Scheduling
    initial_lr = 1e-3
    epochs = 1000
    
    # Cosine decay: smoothly reduces LR from initial to final over all epochs
    # This works much better for oscillating losses than ReduceLROnPlateau
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=epochs * 4,  # 4 batches per epoch approx
        alpha=0.01  # Final LR = initial_lr * 0.01 = 1e-5
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model = NeuralODE(dt=dt, n_steps=steps - 1)
    model.compile(optimizer=optimizer, loss="mse", metrics="accuracy")
    
    print("Training with Cosine LR decay...")
    print(f"LR will decay from {initial_lr} to {initial_lr * 0.01} over {epochs} epochs")
    
    # Simple training - no conflicting callbacks
    history = model.fit(
        y0, y_true, 
        epochs=epochs, 
        batch_size=32,
        verbose=1, 
        validation_split=0.1
    )
    
    # Save model
    model.save('NeuralODE.keras')
    print("Model saved to NeuralODE.keras")
    
    # Plot
    pred_y = model.predict(y0_test)
    y_true_transposed_test = integrate_with_scan(fhn_dynamics, y0_test, dt, steps - 1)
    y_true_test = jnp.transpose(y_true_transposed_test, (1, 0, 2))

    y_true_np = np.array(y_true_test)
    t_np = np.array(t_span)
    
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Phase plot (Sample 0)
    ax[0].plot(y_true_np[0, :, 0], y_true_np[0, :, 1], 'g--', label="True")
    ax[0].plot(pred_y[0, :, 0], pred_y[0, :, 1], 'b-', label="Neural ODE")
    ax[0].set_title("Phase Portrait (FHN)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Time Series (Sample 0)
    ax[1].plot(t_np, y_true_np[0, :, 0], 'g--', label="v True")
    ax[1].plot(t_np, pred_y[0, :, 0], 'b-', label="v Pred")
    ax[1].set_title("Time Series")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("neural_ode_keras.png")
    print("Saved neural_ode_keras.png")
    plt.show()
