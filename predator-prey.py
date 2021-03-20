import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import fsolve
from math import nan
import ode_solver

def main():
    t = np.linspace(0, 200, 2001)
    b = 0.2

    plot_time_series(t, b)
    plt.show()

    orbit = fsolve(shoot, [0.35, 0.35, 21], predator_prey)
    u0 = orbit[:-1]
    T = orbit[-1]
    print('U0: ', u0)
    print('Period: ', T)

    t = np.linspace(0, T, 101)
    orbit_sol = ode_solver.solve_ode(predator_prey, u0, t, ode_solver.rk4_step, 0.001)
    x = orbit_sol[:, 0]
    y = orbit_sol[:, 1]
    plt.plot(t+114.5, y, t+114.5, x)
    plt.show()

    # plot_phase_portrait(t, b)

    # plot_multiple_ICs(t, b, ((0.1, 0),
    #                          (0.1, 0.2),
    #                          (0.1, 0.5),
    #                          (0.3, 0),
    #                          (0.3, 0.2),
    #                          (0.3, 0.5),
    #                          (0.5, 0),
    #                          (0.5, 0.2),
    #                          (0.5, 0.5)))

    # plot_nullclines(b)

def predator_prey(u, t, b=0.2):
    a = 1
    d = 0.1

    x, y = u

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return np.array((dxdt, dydt))

def plot_time_series(t, b):
    sol = ode_solver.solve_ode(lambda u, t: predator_prey(u, t, b), (0.25, 0.25), t, ode_solver.rk4_step, 0.001)

    x = sol[:, 0]
    y = sol[:, 1]

    plt.plot(t, y, t, x)
    plt.xlabel('time, t')
    plt.legend(['y', 'x'])

def plot_phase_portrait(t, b):
    sol = ode_solver.solve_ode(lambda u, t: predator_prey(u, t, b), (0.25, 0.25), t, ode_solver.rk4_step, 0.001)

    x = sol[:, 0]
    y = sol[:, 1]

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')

def plot_multiple_ICs(t, b, u0s):
    # legend_list = []
    for u0 in u0s:
        sol = ode_solver.solve_ode(lambda u, t: predator_prey(u, t, b), u0, t, ode_solver.rk4_step, 0.001)

        x = sol[:, 0]
        y = sol[:, 1]

        plt.plot(x, y)
        # legend_list.append("(x0, y0) = " + str(u0))

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend(legend_list)

def plot_nullclines(b):
    # x nullcline
    xval = np.linspace(0.1,0.6, 51)
    yval = np.zeros(np.size(xval))
    for (i, x) in enumerate(xval):
        result = root(lambda N: predator_prey((x, N), nan, b)[0], 0.25)
        if result.success:
            yval[i] = result.x
        else:
            yval[i] = nan
    plt.plot(xval, yval)

    # y nullcline
    xval = np.linspace(0.1, 0.6, 51)
    yval = np.zeros(np.size(xval))
    for (i, x) in enumerate(xval):
        result = root(lambda N: predator_prey((x, N), nan, b)[1], 0.25)
        if result.success:
            yval[i] = result.x
        else:
            yval[i] = nan
    plt.plot(xval, yval)

    # plt.legend(['x nullcline', 'y nullcline'])

# G(U0) = u0 - F(u0, T)
def G(F, u0, t0, T):
    # Function solves the ODE and returns the difference between u0 and the solution
    sol = ode_solver.solve_ode(F, u0, [t0, T], ode_solver.rk4_step, 0.01)
    return u0 - sol[-1]

# phase condition is dy/dt=0
def phase_condition(f, u0):
    return np.array([f(u0, 0)[1]]) # return dy/dt at initial conditions

def shoot(U, f):
    # U is in the form of initial guesses for variables and T
    u0 = U[:-1]
    T = U[-1]
    return np.concatenate((G(f, u0, 0, T), phase_condition(f, u0)))

if __name__ == "__main__":
    main()
