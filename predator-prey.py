import numpy as np
import matplotlib.pyplot as plt
import shooting
import simulate_ode

def predator_prey(u, t, a=1.0, b=0.2, d=0.1):
    """
    Predator-prey system of equations. Returns dx/dt and dy/dt.

        Parameters:
            u (tuple):   vector in the form (x, y)
            t (float):   value for time
            a (float):   optional, parameter in equations
            b (float):   optional, parameter in equations
            d (float):   optional, parameter in equations

        Returns:
            numpy array of (dx/dt, dy/dt)
    """
    x, y = u

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return np.array((dxdt, dydt))

t = np.linspace(0, 200, 2001)

# Plot time series
simulate_ode.plot_time_series(predator_prey, (0.25, 0.25), t, labels=['x', 'y'])

# Plot phase portrait
simulate_ode.plot_phase_portrait(predator_prey, (0.25, 0.25), t)

# Plot nullclines
# x nullcline
simulate_ode.plot_nullcline(predator_prey, 0.25, (0.1, 0.6), index=0, points=51, show=False)
# y nullcline
simulate_ode.plot_nullcline(predator_prey, 0.25, (0.1, 0.6), index=1, points=51, show=False)

plt.legend(['x', 'y'])
plt.show()

# Find limit cycle through numerical shooting
u0, T = shooting.find_limit_cycle(predator_prey, [0.35, 0.35], 21)
print('U0: ', u0)
print('Period: ', T)

# Plot limit cycle
t = np.linspace(0, T, 101)
# Plot limit cycle as time series
simulate_ode.plot_time_series(predator_prey, u0, t, labels=["x", "y"])
# Plot limit cycle as phase portrait
simulate_ode.plot_phase_portrait(predator_prey, u0, t)