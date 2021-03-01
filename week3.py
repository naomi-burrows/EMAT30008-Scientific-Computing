import ode_solver
import numpy as np
import matplotlib.pyplot as plt
import time

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
print(f'RK4   approximation (h = {h}) of x(1) = {sol}\n')

# Find the Euler and RK4 approximations for x(1) for different values of h
h = np.logspace(-5, 0, 300)
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

# Find step sizes for Euler and RK4 that give the same error
err_to_find = 2.5*10**(-5)
Euler_h = np.interp(err_to_find, err_euler, h)
RK4_h = np.interp(err_to_find, err_rk4, h)
print(f'h values for error of {err_to_find}:')
print(f'    Euler h = {Euler_h}')
print(f'    RK4   h = {RK4_h}\n')

# Time each method with the step size that gives error above
print(f'Time for h values that give error of {err_to_find}:')
# Euler
time0 = time.perf_counter()
ode_solver.solve_ode(f, x0, t, ode_solver.euler_step, Euler_h)
print(f'    Euler : {time.perf_counter() - time0}s')
# RK4
time0 = time.perf_counter()
ode_solver.solve_ode(f, x0, t, ode_solver.rk4_step, RK4_h)
print(f'    RK4   : {time.perf_counter() - time0}s')