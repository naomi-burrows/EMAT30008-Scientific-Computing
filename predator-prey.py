import numpy as np
import matplotlib.pyplot as plt
import ode_solver

def predator_prey(u, t, b):
    a = 1
    d = 0.1

    x = u[0]
    y = u[1]

    dxdt = x*(1-x) - a*x*y/(d+x)
    dydt = b*y*(1-(y/x))

    return np.array((dxdt, dydt))

b0 = 0.1
b1 = 0.5

b = np.linspace(b0, b1, 5)
t = np.linspace(0, 150, 151)

legend_list = []
for bvalue in b:
    sol = ode_solver.solve_ode(lambda u, t: predator_prey(u, t, bvalue), (0.25, 0.25), t, ode_solver.rk4_step, 0.001)

    x = sol[:, 0]
    y = sol[:, 1]

    plt.plot(t, y)
    legend_list.append("b = "+str(round(bvalue, 2)))
plt.legend(legend_list)
plt.show()
