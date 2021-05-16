import numpy as np
from scipy.optimize import fsolve
import shooting

def natural_parameter_continuation(f, u0_, alpha_range, delta_alpha, discretisation=False):
    """
    Solves equation denoted by f for a range of alpha values by natural parameter continuation.

        Parameters:
            f (function):               equation to be solved by natural parameter continuation
            u0_ (float):                initial estimation of the solution at first alpha value
            alpha_range (tuple):        range of alpha values to solve over, (start value, end value)
            delta_alpha (float):        alpha step size
            discretisation (function):  the discretisation to use, e.g. shooting

        Returns:
            list of alpha values and list of solutions
    """

    # Unpack alpha_range and make an array of alpha values
    alpha_start, alpha_end = alpha_range
    num_alphas = int(abs(alpha_end - alpha_start)/delta_alpha)
    alphas = np.linspace(alpha_start, alpha_end, num_alphas)

    # Solve for alpha0
    u0 = fsolve(f, u0_, args=alpha_start)

    # Solve for other alpha values
    us = [u0]
    for alpha in alphas[1:]:
        # solve using previous u value as an initial guess
        us.append(fsolve(f, us[-1], args=alpha)[0])

    return alphas, us

