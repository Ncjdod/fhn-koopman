import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Ground Truth (FHN) ---
FHN_A = 0.7
FHN_B = 0.8
FHN_TAU = 12.5
FHN_I_EXT = 0.5

def fhn_dynamics(y, a=FHN_A, b=FHN_B, tau=FHN_TAU, I_ext=FHN_I_EXT):
    """FitzHugh-Nagumo dynamics. Shape: y is [..., 2]"""
    v = y[..., 0]
    w = y[..., 1]
    dv = v - (v**3)/3 - w + I_ext
    dw = (v + a - b*w) / tau
    return jnp.stack([dv, dw], axis=-1)

# --- 2. Solver (RK4) using JAX ---
def rk4_step(func, y, dt):
    k1 = func(y)
    k2 = func(y + dt*k1/2)
    k3 = func(y + dt*k2/2)
    k4 = func(y + dt*k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def integrate_with_scan(func, y0, dt, n_steps):
    def step_fn(carry, _):
        y = carry
        y_next = rk4_step(func, y, dt)
        return y_next, y_next
    
    _, ys = lax.scan(step_fn, y0, None, length=n_steps)
    ys = jnp.concatenate([y0[None, ...], ys], axis=0)
    return ys

# --- 3. Physics-Informed Neural ODE Model ---
# Using add_loss() approach - simpler and compatible with JAX backend
@keras.saving.register_keras_serializable()
class PhysicsInformedNeuralODE(keras.Model):
    def __init__(self, dt, n_steps, physics_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.n_steps = n_steps
        self.physics_weight = physics_weight
        
        # MLP for vector field
        self.net = keras.Sequential([
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(64, activation="tanh"),
            keras.layers.Dense(2)
        ])

    def call(self, y0, training=False):
        """Forward pass with physics loss added during training."""
        def neural_dynamics(y):
            return self.net(y)
        
        # Integrate trajectory
        ys = integrate_with_scan(neural_dynamics, y0, self.dt, self.n_steps)
        output = jnp.transpose(ys, (1, 0, 2))  # [batch, time, 2]
        
        # Add physics loss during training
        if training:
            phys_loss = self._compute_physics_loss(y0)
            self.add_loss(self.physics_weight * phys_loss)
        
        return output
    
    def _compute_physics_loss(self, y0):
        """
        Enhanced physics loss with:
        1. Random collocation points (general coverage)
        2. Points near the nullclines (where fast dynamics happen)
        3. Initial conditions (anchor points)
        """
        batch_size = y0.shape[0]
        key = jax.random.PRNGKey(42)
        
        # ===== Part 1: Random collocation points =====
        key, key_v, key_w = jax.random.split(key, 3)
        v_rand = jax.random.uniform(key_v, (batch_size,), minval=-2.0, maxval=2.0)
        w_rand = jax.random.uniform(key_w, (batch_size,), minval=-1.0, maxval=1.5)
        y_random = jnp.stack([v_rand, w_rand], axis=-1)
        
        # ===== Part 2: Points near the cubic nullcline (v - v^3/3 - w = 0) =====
        # This is where fast dynamics happen - critical for sharp transitions!
        key, key_v2 = jax.random.split(key, 2)
        v_nullcline = jax.random.uniform(key_v2, (batch_size,), minval=-2.0, maxval=2.0)
        w_nullcline = v_nullcline - (v_nullcline**3)/3 + FHN_I_EXT  # On the nullcline
        # Add small perturbation
        w_nullcline = w_nullcline + jax.random.normal(key, (batch_size,)) * 0.1
        y_nullcline = jnp.stack([v_nullcline, w_nullcline], axis=-1)
        
        # ===== Part 3: Points at extremes (spike peak/trough) =====
        key, key_ext = jax.random.split(key, 2)
        # Sample near v extremes where spikes happen
        v_extreme = jax.random.choice(key_ext, jnp.array([-1.5, 1.5, -2.0, 2.0]), shape=(batch_size,))
        w_extreme = jax.random.uniform(key, (batch_size,), minval=-0.5, maxval=1.0)
        y_extreme = jnp.stack([v_extreme, w_extreme], axis=-1)
        
        # ===== Combine all collocation points =====
        y_all = jnp.concatenate([y0, y_random, y_nullcline, y_extreme], axis=0)
        
        # Network vs true dynamics
        f_net = self.net(y_all)
        f_true = fhn_dynamics(y_all)
        
        physics_loss = jnp.mean((f_net - f_true) ** 2)
        return physics_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "dt": float(self.dt),
            "n_steps": int(self.n_steps),
            "physics_weight": float(self.physics_weight),
        })
        return config

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Parameters
    T = 50.0
    steps = 200
    dt = T / (steps - 1)
    t_span = jnp.linspace(0.0, T, steps)
    
    # Generate Data
    print("Generating ground truth trajectories...")
    key = jax.random.PRNGKey(42)
    n_samples = 121
    key_v, key_w = jax.random.split(key)
    v0 = jax.random.uniform(key_v, (n_samples,), minval=-2.0, maxval=2.0)
    w0 = jax.random.uniform(key_w, (n_samples,), minval=-1.0, maxval=1.5)
    y0_pre = jnp.stack([v0, w0], axis=-1).astype(jnp.float32)
    y0 = y0_pre[:n_samples - 1]
    y0_test = y0_pre[n_samples - 1:n_samples]
    print(f"Generated {n_samples} initial conditions")
    
    # Generate ground truth
    y_true_transposed = integrate_with_scan(fhn_dynamics, y0, dt, steps - 1)
    y_true = jnp.transpose(y_true_transposed, (1, 0, 2))
    print(f"Ground truth shape: {y_true.shape}")

    # Train with Physics-Informed Loss
    initial_lr = 1e-3
    epochs = 1000
    physics_weight = 0.5  # Balance between trajectory and physics loss
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=epochs * 4,
        alpha=0.01
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model = PhysicsInformedNeuralODE(dt=dt, n_steps=steps - 1, physics_weight=physics_weight)
    model.compile(optimizer=optimizer, loss="mse")
    
    print(f"Training Physics-Informed Neural ODE (lambda={physics_weight})...")
    print(f"Total loss = MSE_trajectory + {physics_weight} * MSE_physics")
    print(f"LR will decay from {initial_lr} to {initial_lr * 0.01} over {epochs} epochs")
    
    history = model.fit(
        y0, y_true, 
        epochs=epochs, 
        batch_size=32,
        verbose=1, 
        validation_split=0.1
    )
    
    # Save model
    model.save('PhysicsNeuralODE.keras')
    print("Model saved to PhysicsNeuralODE.keras")
    
    # Plot
    pred_y = model.predict(y0_test)
    y_true_transposed_test = integrate_with_scan(fhn_dynamics, y0_test, dt, steps - 1)
    y_true_test = jnp.transpose(y_true_transposed_test, (1, 0, 2))

    y_true_np = np.array(y_true_test)
    t_np = np.array(t_span)
    
    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    # Phase plot
    ax[0].plot(y_true_np[0, :, 0], y_true_np[0, :, 1], 'g--', linewidth=2, label="True")
    ax[0].plot(pred_y[0, :, 0], pred_y[0, :, 1], 'b-', linewidth=2, label="PINN")
    ax[0].set_title("Phase Portrait (FHN)")
    ax[0].set_xlabel("v")
    ax[0].set_ylabel("w")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Time Series
    ax[1].plot(t_np, y_true_np[0, :, 0], 'g--', linewidth=2, label="v True")
    ax[1].plot(t_np, pred_y[0, :, 0], 'b-', linewidth=2, label="v PINN")
    ax[1].set_title("Time Series")
    ax[1].set_xlabel("Time")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    # Loss history
    ax[2].semilogy(history.history['loss'], label='Total Loss')
    if 'val_loss' in history.history:
        ax[2].semilogy(history.history['val_loss'], label='Val Loss', alpha=0.7)
    ax[2].set_title("Training History")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss (log)")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pinn_neural_ode.png")
    print("Saved pinn_neural_ode.png")
    plt.show()
