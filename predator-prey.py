import numpy as np
import matplotlib.pyplot as plt
import ode_solver

def main():
    t = np.linspace(0, 150, 1501)
    b = 0.3

    # plot_time_series(t, b)

    # plot_phase_portrait(t, b)

    plot_multiple_ICs(t, b, ((0.1, 0),
                             (0.1, 0.2),
                             (0.1, 0.5),
                             (0.3, 0),
                             (0.3, 0.2),
                             (0.3, 0.5),
                             (0.5, 0),
                             (0.5, 0.2),
                             (0.5, 0.5)))

def predator_prey(u, t, b):
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
    plt.show()

def plot_phase_portrait(t, b):
    sol = ode_solver.solve_ode(lambda u, t: predator_prey(u, t, b), (0.25, 0.25), t, ode_solver.rk4_step, 0.001)

    x = sol[:, 0]
    y = sol[:, 1]

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_multiple_ICs(t, b, u0s):
    legend_list = []
    for u0 in u0s:
        sol = ode_solver.solve_ode(lambda u, t: predator_prey(u, t, b), u0, t, ode_solver.rk4_step, 0.001)

        x = sol[:, 0]
        y = sol[:, 1]

        plt.plot(x, y)
        legend_list.append("(x0, y0) = " + str(u0))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(legend_list)
    plt.show()

if __name__ == "__main__":
    main()
