import numpy as np

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
def boris_push(xm, v, q, m, dt, t):
    """
    Advance velocity v by one full step using Boris scheme.
    xm: position at half step (x_{n+1/2})
    v : velocity at time step n
    q, m: charge and mass
    dt: timestep
    t: current time
    """
    E = electric_field(xm, t)
    B = magnetic_field(xm, t)

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
    xs = [x0]
    vs = [v0]

    for n in range(nsteps):
        # Velocity update (Boris)
        v = boris_push(x_half, v, q, m, dt, n * dt)

        # Position update
        x_half = x_half + v * dt
        x = x_half - 0.5 * dt * v  # bring back to integer time

        # Store
        xs.append(x.copy())
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
    nsteps = 100

    # Initial conditions
    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    xs, vs = simulate(x0, v0, q, m, dt, nsteps)

    # Quick check
    import matplotlib.pyplot as plt
    plt.plot(xs[:,0], xs[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Boris pusher trajectory")
    plt.axis("equal")
    plt.show()
