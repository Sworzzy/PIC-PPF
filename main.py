import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Define fields
# -------------------------------------------------
def electric_field(x, t):
    """Return E(x,t) as a numpy array"""
    return np.array([0.0, 0.0, 0.0])  # Example: zero field

def magnetic_field(x, t):
    """Return B(x,t) as a numpy array"""
    return np.array([0.0, 0.0, 1.0])  # Example: uniform Bz field


# -------------------------------------------------
# Boris pusher
# -------------------------------------------------
def boris_push(x, t, v, q, m, dt):
    """
    Advance velocity v by one full step using Boris scheme.
    x: position at half step (x_{n+1/2})
    t: time at half step (t_{n+1/2})
    v: velocity at time step n
    q, m: charge and mass
    dt: timestep
    """
    E = electric_field(x, t)
    B = magnetic_field(x, t)

    # Half acceleration by E
    v_minus = v + (q * dt / (2 * m)) * E

    # Rotation by B
    t_vec = (q * dt / (2 * m)) * B
    t_mag2 = np.dot(t_vec, t_vec)
    s_vec = 2 * t_vec / (1 + t_mag2)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Second half acceleration by E
    v_new = v_plus + (q * dt / (2 * m)) * E

    return v_new

# -------------------------------------------------
# Plot trajectory
# -------------------------------------------------
def plot_trajectory(xs, vs, step=5):
    """
    Plot the particle trajectory with velocity arrows.
    
    xs : array of shape (nsteps+1, 3), positions
    vs : array of shape (nsteps+1, 3), velocities
    step : spacing between arrows (default every 5th point)
    """
    fig, ax = plt.subplots(figsize=(6,6))

    # Plot trajectory (projection on x-y plane)
    ax.scatter(xs[:,0], xs[:,1], s=15, c="blue", label="positions")

    # Add velocity arrows
    ax.quiver(
        xs[::step,0], xs[::step,1],   # arrow bases (positions)
        vs[::step,0], vs[::step,1],   # arrow directions (velocities)
        angles='xy', scale_units='xy', scale=5, color="red", width=0.004,
        label="velocity"
    )

    # Format
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("Boris Pusher: Trajectory with Velocities")

    plt.show()


# -------------------------------------------------
# Main particle loop
# -------------------------------------------------
def simulate(x0, v0, q, m, dt, nsteps):
    """
    Simulate a particle trajectory using the Boris scheme.
    Returns arrays of x and v.
    """
    # Initializations
    x_half = x0 + 0.5 * dt * v0   # Leapfrog staggering
    v = v0.copy()

    # Storage
    xs = [x_half]
    vs = [v0]

    for n in range(nsteps):
        # Velocity update (Boris)
        v = boris_push(x_half, (n + 1/2) * dt, v, q, m, dt)

        # Position update
        x_half = x_half + v * dt

        # Store
        xs.append(x_half.copy())
        vs.append(v.copy())

    return np.array(xs), np.array(vs)


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    # Particle parameters
    q = -1.0   # charge
    m = 1.0    # mass
    dt = 0.1
    nsteps = int(20*np.pi)

    # Initial conditions
    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    xs, vs = simulate(x0, v0, q, m, dt, nsteps)

    plot_trajectory(xs, vs, step=5)
