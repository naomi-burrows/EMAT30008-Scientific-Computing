import numpy as np
import matplotlib.pyplot as plt
import ode_solver

def main():
    t = np.linspace(0, 1000, 1001)
    b = 0.3
    plot_time_series(t, b)

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

if __name__ == "__main__":
    main()
