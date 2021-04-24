import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import shooting

def natural_parameter_continuation(f, u0_, alpha_range, delta_alpha):
    """
    Solves equation denoted by f for a range of alpha values by natural parameter continuation.

        Parameters:
            f (function):           equation to be solved by natural parameter continuation
            u0_ (float):            initial estimation of the solution at first alpha value
            alpha_range (tuple):    range of alpha values to solve over, (start value, end value)
            delta_alpha (float):    alpha step size

        Returns:
            list of alpha values and list of solutions
    """

    # Unpack alpha_range and make an array of alpha values
    alpha_start, alpha_end = alpha_range
    num_alphas = int((alpha_end - alpha_start)/delta_alpha)
    alphas = np.linspace(alpha_start, alpha_end, num_alphas)

    # Solve for alpha0
    u0 = fsolve(f, u0_, args=alpha_start)

    # Solve for other alpha values
    us = [u0]
    for alpha in alphas[1:]:
        # solve using previous u value as an initial guess
        us.append(fsolve(f, us[-1], args=alpha)[0])

    return alphas, us


def cubic_eq(x, c):
    return x**3 -x +c

# Solve by natural parameter continuation
cs, xs = natural_parameter_continuation(cubic_eq, 1.5, (-2, 2), 0.001)

# Plot results - x against c
plt.plot(cs, xs)
plt.show()

# Plot results - f(x) against c, should all be zero
plt.plot(cs, cubic_eq(np.array(xs), np.array(cs)))
plt.show()
