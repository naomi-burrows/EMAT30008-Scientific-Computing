import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import fsolve
from math import nan
import ode_solver
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

simulate_ode.plot_time_series(predator_prey, (0.25, 0.25), t, labels=['x', 'y'])

simulate_ode.plot_phase_portrait(predator_prey, (0.25, 0.25), t)

# x nullcline
simulate_ode.plot_nullcline(predator_prey, 0.25, (0.1, 0.6), index=0, points=51, show=False)
# y nullcline
simulate_ode.plot_nullcline(predator_prey, 0.25, (0.1, 0.6), index=1, points=51, show=False)

plt.legend(['x', 'y'])
plt.show()

u0, T = shooting.find_limit_cycle(predator_prey, [0.35, 0.35], 21)
print('U0: ', u0)
print('Period: ', T)

t = np.linspace(0, T, 101)
orbit_sol = ode_solver.solve_ode(predator_prey, u0, t, ode_solver.rk4_step, 0.001)
x = orbit_sol[:, 0]
y = orbit_sol[:, 1]
plt.plot(t, y, t, x)
plt.show()
