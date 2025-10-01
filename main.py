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

# --- analytic solution for uniform Bz, E=0 ---
def cyclotron_xy(x0, v0, q, m, Bz, times):
    """
    Returns (x_analytic, v_analytic) arrays for times (1D array).
    x_analytic shape: (len(times), 3)
    v_analytic shape: (len(times), 3)
    Only xy components are nontrivial (z preserved from initial).
    """
    omega = (q * Bz) / m
    x_analytic = np.zeros((len(times), 3))
    v_analytic = np.zeros((len(times), 3))

    # initial z components pass through unchanged (since B only in z)
    x_analytic[:,2] = x0[2] + v0[2] * times
    v_analytic[:,2] = v0[2]

    x0_xy = x0[:2].copy()
    v0_xy = v0[:2].copy()

    for i in range(len(times)) :

        x_analytic[i, 0] = x0[0] * np.cos(omega * times[i]) + (v0[0]/omega) * np.sin(omega * times[i])
        x_analytic[i, 1] = x0[1] * np.cos(omega * times[i]) + (v0[1]/omega) * np.sin(omega * times[i])

    return x_analytic, v_analytic


# --- improved plotting comparing numeric and analytic ---
def plot_compare(xs_num, vs_num, x_an, v_an, step=8, show_arrows=True):
    """
    Plot numeric scatter vs analytic curve (xy-projection), plus velocity arrows.
    xs_num: numeric positions shape (N,3)
    vs_num: numeric velocities shape (N,3)
    x_an, v_an: analytic arrays of same shape
    step: sampling period for arrows
    """
    fig, ax = plt.subplots(figsize=(7,7))

    # Numeric scatter
    ax.scatter(xs_num[:,0], xs_num[:,1], s=20, c='C0', label='numeric (Boris)')

    # Analytic curve
    ax.scatter(x_an[:,0], x_an[:,1], s=20, c='C1', label='analytic')

    # initial points markers
    ax.scatter(xs_num[0,0], xs_num[0,1], marker='o', s=60, facecolors='none', edgecolors='k', label='start')

    if show_arrows:
        # numeric arrows (sampled)
        ax.quiver(
            xs_num[::step,0], xs_num[::step,1],
            vs_num[::step,0], vs_num[::step,1],
            angles='xy', scale_units='xy', scale=8, width=0.004,
            label='numeric v', alpha=0.9
        )
        # analytic arrows (sampled, slightly shifted so they are visible)
        ax.quiver(
            x_an[::step,0], x_an[::step,1],
            v_an[::step,0], v_an[::step,1],
            angles='xy', scale_units='xy', scale=8, width=0.006,
            label='analytic v', alpha=0.6
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_title('Numeric (scatter) vs Analytic (line) - xy projection')

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
# your particle parameters
    q = -1.0
    m = 1.0
    dt = 0.1
    nsteps = int(20 * np.pi)  # ~62
    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    Bz = 1.0  # ensure matches the magnetic_field used in simulate()

    # simulate numeric
    xs_num, vs_num = simulate(x0, v0, q, m, dt, nsteps)

    # times corresponding to stored numeric points:
    # recall simulate stored x at integer times from 0..nsteps
    times = (np.arange(len(xs_num))+0.5) * dt

    # analytic
    x_an, v_an = cyclotron_xy(x0, v0, q, m, Bz, times)

    # plot compare
    plot_compare(xs_num, vs_num, x_an, v_an, step=8, show_arrows=True)

    # compute and print RMS error on xy positions
    diff = xs_num[:, :2] - x_an[:, :2]
    rms = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    print(f"RMS position error (xy): {rms:.3e}")

    # optionally show per-step errors
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    errs = np.sqrt(np.sum(diff**2, axis=1))
    ax.plot(times, errs)
    ax.set_xlabel('time')
    ax.set_ylabel('position error (xy)')
    ax.set_title('Error vs time')
    plt.show()
