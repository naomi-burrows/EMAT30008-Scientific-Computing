import ode_solver
import numpy as np
import matplotlib.pyplot as plt

# xdot = x
def f(x,t):
    return x

x0 = [1.]
t = [0.,1.]

# Estimate x(1) - Euler
h = 0.001
sol = ode_solver.solve_ode(f, x0, t, ode_solver.euler_step, h)[1][0]
print(f'Euler approximation (h = {h}) of x(1) = {sol}')

# Estimate x(1) - RK4
h = 0.001
sol = ode_solver.solve_ode(f, x0, t, ode_solver.rk4_step, h)[1][0]
print(f'RK4   approximation (h = {h}) of x(1) = {sol}')

# Find the Euler and RK4 approximations for x(1) for different values of h
h = np.logspace(-4, -1, 300)
real = np.exp(1)    # real value of x(1)
sol_euler = [ode_solver.solve_ode(f, x0, t, ode_solver.euler_step, hvalue)[1][0] for hvalue in h]
sol_rk4 = [ode_solver.solve_ode(f, x0, t, ode_solver.rk4_step, hvalue)[1][0] for hvalue in h]

# Find the error of the Euler and RK4 estimations
err_euler = abs(sol_euler - real)
err_rk4 = abs(sol_rk4 - real)

# Plot the errors on a double logarithmic graph
plt.loglog(h, err_euler, h, err_rk4, linewidth=1.5)
plt.xlabel('Timestep, h')
plt.ylabel('Absolute Error')
plt.legend(['Euler', 'RK4'], loc='best')
plt.savefig('Euler-RK4-Errors.png')
plt.show()
