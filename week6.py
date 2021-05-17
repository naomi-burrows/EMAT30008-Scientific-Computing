import numpy as np
import matplotlib.pyplot as plt
from numerical_continuation import natural_parameter_continuation
import shooting

def cubic_eq(x, c):
    return x**3 -x +c

def hopf(u, t, beta):
    """
    ODE for the Hopf bifurcation normal form. Returns du1/dt and du2/dt

        Parameters:
            u (tuple):      vector in the form (u1, u2)
            t (float):      value for time
            beta (float):   parameter in equations

        Returns:
            numpy array of (du1/dt, du2/dt)
    """
    u1, u2 = u

    du1dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 - u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))


problem_to_run = 2  # 1 for cubic_eq, 2 for hopf, 3 for hopf_mod
if problem_to_run == 1: # cubic_eq code

    # Solve by natural parameter continuation
    cs, xs = natural_parameter_continuation(cubic_eq, 1.5, (2, -2), 0.001, discretisation= lambda u0_, f, alpha: f(u0_, alpha))

    # Plot results - x against c
    plt.plot(cs, xs)
    plt.show()

    # Plot results - f(x) against c, should all be zero
    plt.plot(cs, cubic_eq(np.array([x[0] for x in xs]), np.array(cs)))
    plt.show()

elif problem_to_run == 2: # hopf code

    # Solve by natural parameter continuation
    betas, us = natural_parameter_continuation(hopf, (-1, 0, 6), (0, 2), 0.01, discretisation=lambda u0_, f, alpha: shooting.shoot(u0_, f, 0, alpha))

    # Plot results - u1 and u2 against beta
    plt.plot(betas, [u[0] for u in us], betas, [u[1] for u in us])
    plt.legend(['u1', 'u2'])
    plt.show()


