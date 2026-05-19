import numpy as np
import jax
import jax.numpy as jnp

def hamiltonian(state):
    x, y, p_x, p_y = state
    H = (1/2) * (p_x**2 + p_y**2) + (1/2) * (x**2 + y**2) + x**2 * y - y**3 / 3
    return H

def equations_of_motion(state):
    dh_dx, dh_dy, dh_dpx, dh_dpy = jax.grad(hamiltonian)(state)
    return jnp.array([dh_dpx, dh_dpy, -dh_dx, -dh_dy])


def runge_kutta(state, dt):
    k1 = equations_of_motion(state)
    k2 = equations_of_motion(state + dt * k1 / 2)
    k3 = equations_of_motion(state + dt * k2 / 2)
    k4 = equations_of_motion(state + dt * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


@jax.jit(static_argnums=(1, 2))
def solve(state, dt, t_max):
    num_steps = int(t_max / dt)
    
    def step(carry, _):
        current_state = carry
        multiplier = runge_kutta(current_state, dt)
        next_state = current_state + dt * multiplier
        return next_state, next_state

    # Use scan to run the loop efficiently on the XLA device
    final_state, history = jax.lax.scan(step, state, None, length=num_steps)
    
    # Prepend the initial state to the history
    history = jnp.vstack([state, history])
    
    return history


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_orbit(history, interval=10):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Henon-Heiles System Orbit')
    
    line, = ax.plot([], [], 'b-', lw=0.5, alpha=0.8)
    point, = ax.plot([], [], 'ro', markersize=4)
    
    # Extract x and y coordinates from history
    # history shape is (num_steps + 1, 4), where columns are x, y, px, py
    xs = history[:, 0]
    ys = history[:, 1]
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def update(frame):
        # Draw trail up to current frame
        line.set_data(xs[:frame], ys[:frame])
        # Draw current position
        point.set_data([xs[frame]], [ys[frame]])
        return line, point
    
    ani = FuncAnimation(fig, update, frames=len(history),
                        init_func=init, blit=True, interval=interval / 10)
    plt.show()
    return ani

if __name__ == "__main__":
    # Example usage
    # Energy E = 1/12 is a critical value. Try slightly below, e.g., E ~ 0.0833
    # Initial state: [x, y, px, py]
    # Let's pick x=0, px=some_val, y=0.1, py=0 to get a specific energy
    initial_state = jnp.array([0.0, 0.1, 0.4, 0.0]) 
    dt = 0.05
    t_max = 200.0
    
    print("Solving system...")
    trajectory = solve(initial_state, dt, t_max)
    print(f"Simulation complete. Steps: {len(trajectory)}")
    
    animate_orbit(trajectory)
