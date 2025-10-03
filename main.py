import numpy as np
import matplotlib.pyplot as plt
from plot import plot_trajectory, plot_compare, plot_errors
from plasma import cyclotron_xy


# -------------------------------------------------
# Define fields
# -------------------------------------------------
def electric_field(x, t):
    """Return E(x,t) as a numpy array"""
    return np.array([0.0, 1.0, 0.0])  # Example: zero field

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
# Time loop
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
# Main
# -------------------------------------------------
if __name__ == "__main__":

    q = -1.0
    m = 1.0
    B0 = 1.0

    omega = q*B0/m
    T=2*np.pi/abs(omega)
    dt = 0.1
    NT = 2
    nsteps = int(NT*T/dt)

    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])


    # Numerical trajectory
    xs_num, vs_num = simulate(x0, v0, q, m, dt, nsteps)
    
    plot_trajectory(xs_num, vs_num, step=5)

    # # Analytical trajectory : cyclotron motion
    # times = np.arange(0, T, dt) +dt/2
    # x_an, v_an = cyclotron_xy(x0, v0, q, m, B0, times)


    # # RMS Error on xy positions
    # diff = xs_num[:, :2] - x_an[:, :2]
    # rms = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    # print(f"RMS position error (xy): {rms:.3e}")


    # Plots to compare
    # plot_compare(xs_num, vs_num, x_an, v_an, step=8, show_arrows=True)
    # plot_errors(times, diff)


