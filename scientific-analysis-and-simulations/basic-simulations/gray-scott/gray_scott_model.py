import jax
import jax.numpy as jnp

class Gray_scott():

    def __init__(self, D_prey, D_predator, feed_rate, kill_rate, grid_size=100):
        self.size = grid_size
        self.u = jnp.ones((grid_size, grid_size))
        self.v = jnp.zeros((grid_size, grid_size))
        
        mid = grid_size // 2
        r = 10
        self.u = self.u.at[mid-r:mid+r, mid-r:mid+r].set(0.50)
        self.v = self.v.at[mid-r:mid+r, mid-r:mid+r].set(0.25)
        
        self.state = jnp.stack([self.u, self.v])
        
        self.Du = D_prey
        self.Dv = D_predator
        self.F = feed_rate
        self.k = kill_rate

    @staticmethod
    def laplacian(grid):
        kernel = jnp.array([[0, 1, 0], 
                        [1, -4, 1],
                        [0, 1, 0]])
        return jax.scipy.signal.convolve2d(grid, kernel, mode='same')
    
    def simulation(self):
        Du, Dv = self.Du, self.Dv
        F, k = self.F, self.k
        
        @jax.jit
        def step_fn(state, _):
            u, v = state[0], state[1]
            
            lu = Gray_scott.laplacian(u)
            lv = Gray_scott.laplacian(v)
            
            reaction = u * (v**2)
            
            du = (Du * lu) - reaction + (F * (1 - u))
            dv = (Dv * lv) + reaction - ((F + k) * v)
            
            d_state = jnp.stack([du, dv])
            
            dt = 1.0
            new_state = state + (dt * d_state)
            new_state = jnp.clip(new_state, 0, 1)
            
            return new_state, new_state

        num_steps = 0
        xs = jnp.arange(num_steps)
        init_state = self.state

        final_state, _ = jax.lax.scan(step_fn, init_state, xs)

        U_final, V_final = final_state[0], final_state[1]

        xs_ani = jnp.arange(3000)
        final_state_animation, history = jax.lax.scan(step_fn, final_state, xs_ani)
        
        U_history = history[:, 0]
        V_history = history[:, 1]
        
        return U_final, V_final, U_history, V_history


model = Gray_scott(0.16, 0.08, 0.060, 0.062)
uf, vf, uh, vh = model.simulation()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Subsample history for faster animation (e.g., every 10th step)
step = 10
uh_anim = uh[::step]
vh_anim = vh[::step]

print(f"Creating animation with {len(uh_anim)} frames...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Initial Plots
im1 = ax1.imshow(uh_anim[0], cmap='GnBu', vmin=0, vmax=1)
ax1.set_title("Prey (U)")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(vh_anim[0], cmap='inferno', vmin=0, vmax=1)
ax2.set_title("Predator (V)")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

def update(frame_idx):
    im1.set_array(uh_anim[frame_idx])
    im2.set_array(vh_anim[frame_idx])
    return [im1, im2]

ani = animation.FuncAnimation(fig, update, frames=len(uh_anim), blit=True, interval=30)

try:
    ani.save('gray_scott.gif', writer='pillow', fps=30)
    print("Animation saved as 'gray_scott.gif'")
except Exception as e:
    print(f"Could not save GIF: {e}")
plt.show()