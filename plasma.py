import numpy as np



# -------------------------------------------------
# Analytic solution for uniform Bz0, E=0
# -------------------------------------------------
def cyclotron_xy(x0, v0, q, m, Bz, times, dt):
    """
    Returns (x_analytic, v_analytic) arrays for times (1D array).
    x_analytic shape: (len(times), 3)
    v_analytic shape: (len(times), 3)
    Only xy components are nontrivial (z preserved from initial).
    """
    omega = (q * Bz) / m
    x_analytic = np.zeros((len(times), 3))
    v_analytic = np.zeros((len(times), 3))

    x_analytic[:,2] = x0[2] + v0[2] * times
    v_analytic[:,2] = v0[2]

    x0_xy = x0[:2].copy()
    v0_xy = v0[:2].copy()

    for i in range(len(times)) :

        x_analytic[i, 0] = x0[0] + (v0[0]/omega) * np.sin(omega * times[i]) + (v0[1]/omega) * (1 - np.cos(omega * times[i]))
        x_analytic[i, 1] = x0[1] + (v0[1]/omega) * np.sin(omega * times[i]) - (v0[0]/omega) * (1 - np.cos(omega * times[i]))
        
        v_analytic[i, 0] = v0[0] * np.cos(omega * (times[i] - dt/2) ) + v0[1] * np.sin(omega * (times[i] - dt/2) )
        v_analytic[i, 1] = v0[1] * np.cos(omega * (times[i] - dt/2) ) - v0[0] * np.sin(omega * (times[i] - dt/2) )

    return x_analytic, v_analytic


# -------------------------------------------------
# Analytic solution for E x B : Bz0, Ey0
# -------------------------------------------------
def ExB_xy(x0, v0, q, m, E, B, times, dt):
    """
    E x B drift + cyclotron motion in relative reference frame (vrel = v - vd)
    """
    E = np.asarray(E)
    B = np.asarray(B)

    Bz = B[2]
    # ExB drift
    vd = np.cross(E, B) / (Bz**2)

    v_rel0 = np.asarray(v0) - vd

    x_cyc, v_cyc = cyclotron_xy(x0, v_rel0, q, m, Bz, times, dt)

    v_analytic = v_cyc + vd
    x_analytic = x_cyc + np.outer(times, vd)    # (N,3) + (N,1)*(3,)

    return x_analytic, v_analytic