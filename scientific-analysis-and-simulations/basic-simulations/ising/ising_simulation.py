import numpy as np
import jax
import jax.numpy as jnp

key_in = jax.random.PRNGKey(41)
grid = jax.random.randint(key_in, shape=(100, 100), minval=0, maxval=2)
grid = 2 * grid - 1
temperature = 2.27


def energy_state(grid, J_const=1):
    rolled_grids = {}
    for axis in [0, 1]:
        rolled_grids[axis] = {}
        for direction in [-1, 1]:
            rolled = jnp.roll(grid, direction, axis)
            
            edge_idx = 0 if direction == 1 else -1
            
            if axis == 0:
                rolled = rolled.at[edge_idx, :].set(0)
            else:
                rolled = rolled.at[:, edge_idx].set(0)
                
            rolled_grids[axis][direction] = rolled
    
    energy_grid = J_const * grid * (rolled_grids[0][1] + rolled_grids[1][1] + rolled_grids[0][-1] + rolled_grids[1][-1])
    energy = jnp.sum(energy_grid)
    return energy, energy_grid

@jax.jit
def metropolis_step(grid, key_in, temperature=1):
    new_key, subkey_idx, subkey_prob = jax.random.split(key_in, 3)
    
    # Fix: shape=(2,) for randint to get a pair
    idx = jax.random.randint(subkey_idx, shape=(2,), minval=0, maxval=100)
    r, c = idx[0], idx[1]
    
    # Optimization: Calculate Delta E locally (O(1))
    _, energy_grid = energy_state(grid)
    delta_energy = 2 * energy_grid[r, c]

    # Fix: jax.random.uniform(key) returns scalar by default
    random_val = jax.random.uniform(subkey_prob)
    accept = (delta_energy < 0) | (random_val < jnp.exp(-delta_energy / temperature))
    
    # Fix: Use jax.lax.select
    new_val = grid[r, c] * -1
    grid = grid.at[r, c].set(
        jax.lax.select(accept, new_val, grid[r, c])
    )

    return grid, new_key

# --- Animation Program ---
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def frame_gen(temp=temperature):
    print("Initializing Simulation...")
    current_grid = grid
    current_key = key_in
    
    n_frames = 50
    steps_per_frame = 20000
    
    frames = [current_grid]
    
    @jax.jit
    def run_batch(g, k):
        def loop_body(c, _):
            return metropolis_step(c[0], c[1], temp), None
        (g_final, k_final), _ = jax.lax.scan(loop_body, (g, k), jnp.arange(steps_per_frame))
        return g_final, k_final

    print(f"Simulating {n_frames} frames...")
    for i in range(n_frames):
        current_grid, current_key = run_batch(current_grid, current_key)
        frames.append(current_grid)

    return frames, current_grid


def generate_animation(temp=temperature):
    frames, g_final = frame_gen(temp)
    print("Creating Animation...")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0], cmap='coolwarm', animated=True)
    ax.set_title(f"Ising 2D (T={temp})")
    
    def update_fig(frame):
        im.set_array(frame)
        return [im]
    ani = animation.FuncAnimation(fig, update_fig, frames=frames, blit=True)
    try:
        ani.save('ising.gif', writer='pillow', fps=50)
        print("Done! Saved 'ising.gif'")
    except Exception as e:
        print(f"Animation saved failed: {e}")


def magnetization(grid=grid):
    frames, g_final = frame_gen()
    last_frame = frames[-1]
    N = len(grid.flatten())
    M = jnp.abs(1/N * jnp.sum(last_frame))
    return M


if __name__ == "__main__":
    generate_animation()
