import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------------------
# Plot trajectory of single particle
# -------------------------------------------------

def plot_trajectory_mirror(xs_num,t):
    fig=plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(xs_num[:,2], xs_num[:,1], xs_num[:,0], c=t, cmap='viridis', s=5)  # s controls point size
    ax.scatter(xs_num[:,2], xs_num[:,1], xs_num[:,0], c=t, cmap='viridis', s=5)  # s controls point size
    # plt.colorbar(label='Time')
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    # ax.set_xlim(-2000,2000)
    ax.grid()
    plt.show()

def plot_trajectory(xs, vs, step=5,axis=[0,1]):
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
        xs[::step,axis[0]], xs[::step,axis[1]],   # arrow bases (positions)
        vs[::step,axis[0]], vs[::step,axis[1]],   # arrow directions (velocities)
        angles='xy', scale_units='xy', scale=5, color="red", width=0.004,
        label="velocity"
    )
    for i in range(2):
        if axis[i]==0:
            ax_label="x"
        elif axis[i]==1:
            ax_label="y"
        elif axis[i]==2:
            ax_label="z"
        else:
            ax_label="?"

        if i==0:
            label_x=ax_label
        else:
            label_y=ax_label
    # Format
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    # ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("Boris Pusher: Trajectory with Velocities")

    plt.show()

def plot_trajectory_3d(xs, vs, step=5):
    """
    Plot the particle trajectory in 3D with velocity arrows.

    xs : array of shape (nsteps+1, 3), positions
    vs : array of shape (nsteps+1, 3), velocities
    step : spacing between arrows (default every 5th point)
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(xs[:,0], xs[:,1], xs[:,2], color="blue", label="trajectory")

    # Add velocity arrows
    ax.quiver(
        xs[::step,0], xs[::step,1], xs[::step,2],   # arrow bases (positions)
        vs[::step,0], vs[::step,1], vs[::step,2],   # arrow directions (velocities)
        length=10, normalize=True, color="red", linewidth=1,
        label="velocity"
    )

    # Format
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.set_title("Boris Pusher: 3D Trajectory with Velocities")

    plt.show()
# -------------------------------------------------
# Plot comparing numeric and analytic solutions ---
# -------------------------------------------------
def plot_compare(x_num, vs_num, x_an, v_an, step=8, show_arrows=True):
    """
    Plot numeric scatter vs analytic curve (xy-projection), plus velocity arrows.
    x_num: numeric positions shape (N,3)
    vs_num: numeric velocities shape (N,3)
    x_an, v_an: analytic arrays of same shape
    step: sampling period for arrows
    """
    fig, ax = plt.subplots(figsize=(7,7))

    # Numeric scatter
    ax.scatter(x_num[:,0], x_num[:,1], s=20, c='C0', label='numeric (Boris)')

    # Analytic curve
    ax.scatter(x_an[:,0], x_an[:,1], s=20, c='C1', label='analytic')

    # initial points markers
    ax.scatter(x_num[0,0], x_num[0,1], marker='o', s=60, facecolors='none', edgecolors='k', label='start')

    if show_arrows:
        # numeric arrows (sampled)
        ax.quiver(
            x_num[::step,0], x_num[::step,1],
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

def plot_errors(times, diff) :
    """
    Plot RMS in position over time
    times: instants
    diff: difference of positions x_an - x_num
    """
    fig, ax = plt.subplots()
    errs = np.sqrt(np.sum(diff**2, axis=1))
    ax.plot(times, errs)
    ax.set_xlabel('time')
    ax.set_ylabel('position error (xy)')
    ax.set_title('Error vs time')
    plt.show()

def plot_time_kinetic_energy(times, m, vs_num) :
    v_norm = np.linalg.norm(vs_num, axis=1)
    Ek_x = 0.5 * m * vs_num[:,0]**2
    Ek_y = 0.5* m * vs_num[:,1]**2
    Ek_z = 0.5 * m * vs_num[:,2]**2
    Ek_tot = 0.5 * m * v_norm**2
    plt.figure(figsize=(6,4))
    plt.grid()
    plt.plot(times, Ek_x, label=r'$E_{k,x}$')
    plt.plot(times, Ek_y, label=r'$E_{k,y}$')
    plt.plot(times, Ek_z, label=r'$E_{k,z}$')
    plt.plot(times, Ek_tot, label=r"$E_{k, tot}$", color='black')
    plt.xlabel('time')
    plt.title('Kinetic energy vs time')
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.show()