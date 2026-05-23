import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LorenzAttractor:
    def __init__(self, initial_conditions, num_particles, rho=28, sigma=10, beta=8/3, dt=0.01, t_max=100):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.num_particles = num_particles
        self.dt = dt
        self.t_max = t_max
        
        # Calculate number of steps
        self.n_steps = int(t_max / dt)
        self.time = np.linspace(0, t_max, self.n_steps)
        
        # Pre-allocate arrays
        self.x = np.zeros((self.num_particles, self.n_steps))
        self.y = np.zeros((self.num_particles, self.n_steps))
        self.z = np.zeros((self.num_particles, self.n_steps))
        
        # Set initial conditions
        # Add a small random perturbation to initial conditions for each particle if num_particles > 1
        if num_particles > 1:
            self.x[:, 0] = initial_conditions[0] * np.random.uniform(-10, 10, self.num_particles)
            self.y[:, 0] = initial_conditions[1] * np.random.uniform(-10, 10, self.num_particles)
            self.z[:, 0] = initial_conditions[2] * np.random.uniform(-10, 10, self.num_particles)
        else:
            self.x[:, 0] = initial_conditions[0]
            self.y[:, 0] = initial_conditions[1]
            self.z[:, 0] = initial_conditions[2]

    def eq_of_sys(self, t, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])

    def solve(self):
        # RK4 Integration
        for i in range(self.n_steps - 1):
            current_state = np.array([self.x[:, i], self.y[:, i], self.z[:, i]])
            t = self.time[i]
            
            k1 = self.eq_of_sys(t, current_state)
            k2 = self.eq_of_sys(t + 0.5*self.dt, current_state + 0.5*self.dt*k1)
            k3 = self.eq_of_sys(t + 0.5*self.dt, current_state + 0.5*self.dt*k2)
            k4 = self.eq_of_sys(t + self.dt, current_state + self.dt*k3)
            
            next_state = current_state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            self.x[:, i+1] = next_state[0]
            self.y[:, i+1] = next_state[1]
            self.z[:, i+1] = next_state[2]

        return self.x, self.y, self.z

    def animate(self, interval=10):
        # Ensure we have data
        self.solve()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set limits based on the data to ensure it's visible
        ax.set_xlim((np.min(self.x), np.max(self.x)))
        ax.set_ylim((np.min(self.y), np.max(self.y)))
        ax.set_zlim((np.min(self.z), np.max(self.z)))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Lorenz Attractor ({self.num_particles} particles)')
        
        # Create line and point objects for each particle
        lines = []
        points = []
        # Use a colormap to distinguish particles
        colors = plt.cm.jet(np.linspace(0, 1, self.num_particles))
        
        for i in range(self.num_particles):
            # Initialize empty lines and points
            line, = ax.plot([], [], [], lw=0.5, color=colors[i])
            point, = ax.plot([], [], [], 'o', color=colors[i])
            lines.append(line)
            points.append(point)

        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
            return lines + points

        def update(frame):
            # Update each particle
            for i in range(self.num_particles):
                # Update line (trajectory)
                # We must pass 1D arrays to set_data
                lines[i].set_data(self.x[i, :frame], self.y[i, :frame])
                lines[i].set_3d_properties(self.z[i, :frame])
                
                # Update point (head)
                # Pass sequences (lists or arrays) of length 1
                points[i].set_data([self.x[i, frame]], [self.y[i, frame]])
                points[i].set_3d_properties([self.z[i, frame]])
            
            return lines + points

        print("Creating animation...")
        # Using every 5th frame to speed up rendering
        step = 5
        frames = range(0, self.n_steps, step)
        
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            interval=interval,
            blit=False
        )
        plt.show()
        print("Animation finished.")

def main():
    initial_conditions = np.array([3.0, -3.0, -10.0])
    # You can change num_particles to > 1 to see multiple trajectories
    lorenz = LorenzAttractor(initial_conditions, num_particles=5, t_max=50, dt=0.01)
    lorenz.animate()

if __name__ == '__main__':
    main()