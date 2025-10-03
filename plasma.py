import numpy as np



# -------------------------------------------------
# Analytic solution for uniform Bz, E=0
# -------------------------------------------------
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