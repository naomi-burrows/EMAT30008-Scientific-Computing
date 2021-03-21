import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import fsolve
from math import nan
import ode_solver
import shooting

def main():
    t = np.linspace(0, 200, 2001)
    b = 0.2

    plot_time_series(t, b)
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

if __name__ == "__main__":
    main()
