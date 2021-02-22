import ode_solver
import numpy as np
import matplotlib.pyplot as plt

def f(x,t):
    return x

x0 = [1.,0.]
t = [0.,1.]

h = np.logspace(-9, 0, 30)
err = []
real = np.exp(1)
for hvalue in h:
    estimate = ode_solver.solve_ode(f, x0, t, ode_solver.euler_step, hvalue)
    err.append(abs(estimate[1] - real))

plt.loglog(h, err)
plt.show()
