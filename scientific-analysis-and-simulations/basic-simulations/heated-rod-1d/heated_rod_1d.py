import jax
import jax.numpy as jnp
import numpy as np


class Rod:
    def __init__(self, rod_length, alpha, initial_conditions, boundary_conditions, dt=0.1, t_max=50, length_nodes=100):
        self.rod_length = rod_length
        self.alpha = alpha
        self.num_length_nodes = length_nodes
        self.initial_temp = initial_conditions
        self.dx = self.rod_length / self.num_length_nodes
        self.dt = dt
        self.t_max = t_max
        self.curve_const = (self.alpha * self.dt)/(self.dx**2)
        
        bc1, bc2 = boundary_conditions
        self.bc_left = bc1 if bc1 is not None else jnp.nan
        self.bc_right = bc2 if bc2 is not None else jnp.nan

        self.temp_arr = jnp.array(self.initial_temp)
        

    def simulate(self):
        num_steps = int(self.t_max / self.dt)
        
        bc_vals = jnp.array([self.bc_left, self.bc_right])
        bc_indices = jnp.array([0, -1])

        def step_fn(u, _):
            u_left = jnp.roll(u, 1)
            u_left = u_left.at[0].set(u[0])
            
            u_right = jnp.roll(u, -1)
            u_right = u_right.at[-1].set(u[-1])
            
            u_next = u + self.curve_const * (u_left - 2 * u + u_right)
            
            current_boundary_vals = u_next[bc_indices]
            new_boundary_vals = jnp.where(jnp.isnan(bc_vals), current_boundary_vals, bc_vals)
            
            u_next = u_next.at[bc_indices].set(new_boundary_vals)
            return u_next, u_next

        final_u, history = jax.lax.scan(step_fn, self.temp_arr, None, length=num_steps)
        
        history = jnp.vstack([self.temp_arr, history])
        return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    L = 0.2
    alpha = 23e-6
    nodes = 100
    
    initial_temp = np.full(nodes, 20.0)
    initial_temp[40:60] = 80.0 
    
    bcs = (100.0, None) 
    
    dt = 0.005 
    t_max = 500.0
    
    rod = Rod(L, alpha, initial_temp, bcs, dt=dt, t_max=t_max, length_nodes=nodes)
    print("Running simulation...")
    history = rod.simulate()
    print(f"Simulation done. Shape: {history.shape}")
    
    plt.figure(figsize=(10, 5))
    plt.imshow(history, aspect='auto', cmap='inferno', origin='lower',
               extent=[0, L, 0, t_max])
    plt.colorbar(label='Temperature (°C)')
    plt.xlabel('Position on Rod (x)')
    plt.ylabel('Time (t)')
    plt.title('1D Heat Diffusion: Space-Time Diagram')
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, L)
    min_temp, max_temp = np.min(history), np.max(history)
    ax.set_ylim(min_temp - 5, max_temp + 5)
    
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Rod Temperature Evolution')
    
    line, = ax.plot([], [], 'r-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    x_vals = np.linspace(0, L, nodes)
    
    skip = max(1, len(history) // 300)
    frames = range(0, len(history), skip)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def update(frame_idx):
        temp_profile = history[frame_idx]
        line.set_data(x_vals, temp_profile)
        current_time = frame_idx * dt
        time_text.set_text(f'Time: {current_time:.2f}s')
        return line, time_text
    
    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True, interval=2)
    
    plt.show()