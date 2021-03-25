import numpy as np
import simulate_ode
import shooting

def hopf(u, t, beta, sigma=-1.0):
    """
    ODE for the Hopf bifurcation normal form. Deturns du1/dt and du2/dt

        Parameters:
            u (tuple):      vector in the form (u1, u2)
            t (float):      value for time
            beta (float):   parameter in equations
            sigma (float):  default=-1.0. Parameter in equations

        Returns:
            numpy array of (du1/dt, du2/dt)
    """
    u1, u2 = u

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))

t = np.linspace(0, 40, 401)

# Plot time series
simulate_ode.plot_time_series(lambda u,t: hopf(u, t, beta=1), (1, 1), t, labels=['u1', 'u2'])

# Plot phase portrait
simulate_ode.plot_phase_portrait(lambda u,t: hopf(u, t, beta=1), (1, 1), t, labels=['u1', 'u2'])

# Find limit cycle through numerical shooting
u0, T = shooting.find_limit_cycle(lambda u,t: hopf(u, t, beta=1), (-1, 0), 6)
print('U0: ', u0)
print('Period: ', T)

# Plot limit cycle
t = np.linspace(0, T, 101)
# Plot limit cycle as time series
simulate_ode.plot_time_series(lambda u,t: hopf(u, t, beta=1), u0, t, labels=['u1', 'u2'])
# Plot limit cycle as phase portrait
simulate_ode.plot_phase_portrait(lambda u,t: hopf(u, t, beta=1), u0, t, labels=['u1', 'u2'])