import numpy as np
import matplotlib.pyplot as plt
from plot import plot_trajectory, plot_compare, plot_errors,plot_trajectory_3d
from plasma import cyclotron_xy, ExB_xy
from B_field import MagneticField

# -------------------------------------------------
# Define fields
# -------------------------------------------------


# def electric_field(x, t):
#     """Return E(x,t) as a numpy array"""
#     return np.array([0.0, 1.0, 0.0])  # Example: zero field

def magnetic_field(x, t,B0,B0_space):
    """Return B(x,t) as a numpy array"""
    distances = np.sqrt((B0_space[0] - x[0])**2 + (B0_space[1] - x[1])**2 + (B0_space[2] - x[2])**2)
    flat_index = np.argmin(distances)

    # Convert the flat index to 3D indices
    i, j, k = np.unravel_index(flat_index, B0_space[0].shape)

    return np.array([B0[0][i,j,k], B0[1][i,j,k], B0[2][i,j,k]])
# -------------------------------------------------
# Boris pusher
# -------------------------------------------------
def boris_push(x, t, v, q, m, dt,B0,E0,B0_space):
    """
    Advance velocity v by one full step using Boris scheme.
    x: position at half step (x_{n+1/2})
    t: time at half step (t_{n+1/2})
    v: velocity at time step n
    q, m: charge and mass
    dt: timestep
    """
    # E = electric_field(x, t)
    E=E0
    B = magnetic_field(x, t,B0,B0_space)

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
def simulate(x0, v0, q, m, dt, nsteps, B0, E0,B0_space):
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
        v = boris_push(x_half, (n + 1/2) * dt, v, q, m, dt, B0, E0,B0_space)

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
    NT = 100
    nsteps = int(NT*T/dt)

    x0 = np.array([0.0,0.0, 500.0])
    v0 = np.array([0.0, 100.0, 50.0])
    # v0 = np.array([0.0, 10.0, 100.0])

    # x0=np.array([0

    # E0 = np.array([0.0, 1.0, 0.0])
    # B0 = np.array([0.0, 0.0, 1.0])

    B_field=MagneticField(k=0.1, zc=1000, r=np.linspace(0.1,1500,15), theta=np.linspace(0,2*np.pi,36), z=np.linspace(-2000,2000,25))
    Bx,By,Bz,X,Y,Z=B_field.Compute_field()
    B0=[Bx,By,Bz]
    B0_space=[X,Y,Z]

    E0=np.array([0.0, 0.0, 0.0])
    # B0=np.zeros_like(B0)
   


    # NumericalS trajectory
    xs_num, vs_num = simulate(x0, v0, q, m, dt, nsteps,B0,E0,B0_space)
    print(xs_num)
    plt.figure()
    plt.plot(xs_num[:,2],xs_num[:,1])
    plt.xlabel('z')
    plt.ylabel('y')
    plt.show()

    quit()
    plot_trajectory(xs_num, vs_num, step=5,axis=[2,1])
    plot_trajectory_3d(xs_num, vs_num, step=5)

    # # Analytical trajectory : cyclotron motion
    times = np.arange(0, NT*T, dt) + dt/2
    # x_cyc, v_cyc = cyclotron_xy(x0, v0, q, m, B0[2], times)
    
    x_ExB, v_ExB = ExB_xy(x0, v0, q, m, E0, B0, times)

    # print(vs_num[0,:], v_ExB[0,:])

    # # RMS Error on xy positions
    diff = xs_num[:, :2] - x_ExB[:, :2]
    rms = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    print(f"RMS position error (xy): {rms:.3e}")

    # Plots to compare
    plot_compare(xs_num, vs_num, x_ExB, v_ExB, step=8, show_arrows=True)
    plot_errors(times, diff)
    


