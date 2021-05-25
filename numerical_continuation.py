import numpy as np
from scipy.optimize import fsolve
import math

def natural_parameter_continuation(f, u0_, alpha_range, delta_alpha, discretisation):
    """
    Solves equation denoted by f for a range of alpha values by natural parameter continuation.

        Parameters:
            f (function):               equation to be solved by natural parameter continuation
            u0_ (tuple):                initial estimation of the solution at first alpha value
            alpha_range (tuple):        range of alpha values to solve over, (start value, end value)
            delta_alpha (float):        alpha step size
            discretisation (function):  the discretisation to use, e.g. shooting. Function args: u0_, f, alpha

        Returns:
            list of alpha values and list of solutions
    """

    # Unpack alpha_range and make an array of alpha values
    alpha_start, alpha_end = alpha_range
    num_alphas = math.floor(abs(alpha_end - alpha_start)/delta_alpha)
    alphas = np.linspace(alpha_start, alpha_end, num_alphas)

    # Solve for alpha0
    u0 = fsolve(lambda U, f: discretisation(U, f, alpha_start), u0_, f)

    # Solve for other alpha values
    us = [u0]
    for alpha in alphas[1:]:
        # solve using previous u value as an initial guess
        us.append(fsolve(lambda U, f: discretisation(U, f, alpha), us[-1], f))

    return alphas, us

def pseudo_arclength(f, u0_, alpha_range, delta_alpha, discretisation):
    """
    Solves equation denoted by f for a range of alpha values by pseudo-arclength continuation.

        Parameters:
            f (function):               equation to be solved by natural parameter continuation
            u0_ (tuple):                initial estimation of the solution at first alpha value
            alpha_range (tuple):        range of alpha values to solve over, (start value, end value)
            delta_alpha (float):        alpha step size
            discretisation (function):  the discretisation to use, e.g. shooting. Function args: u0_, f, alpha

        Returns:
            list of alpha values and list of solutions
    """

    # Unpack alpha_range and find the second alpha value, and the stopping condition (alpha_end is reached)
    alpha_start, alpha_end = alpha_range
    if alpha_end - alpha_start < 0:
        alphas = [alpha_start, alpha_start - delta_alpha]
        end_condition = lambda alpha: alpha < alpha_end
    else:
        alphas = [alpha_start, alpha_start + delta_alpha]
        end_condition = lambda alpha: alpha > alpha_end

    # Solve for alpha0
    u0 = fsolve(lambda U, f: discretisation(U, f, alpha_start), u0_, f)

    # Solve for alpha1 using natural parameter continuation
    u1 = fsolve(lambda U, f: discretisation(U, f, alphas[1]), u0, f)

    # Solve until the final alpha value is reached using pseudo-arclength continuation
    us = [u0, u1]
    i = 2
    while not end_condition(alphas[-1]):

        # Generate the secant
        nu0 = np.array([alphas[i-2], *us[i-2]])
        nu1 = np.array([alphas[i-1], *us[i-1]])
        delta = nu1 - nu0

        # Predict solution
        nu2_ = nu1 + delta

        # Pseudo-arclength equation
        PAeq = lambda nu2: np.dot(nu2 - nu2_, delta)

        # Solve
        sol = fsolve(lambda nu2, f: np.append(discretisation(nu2[1:], f, nu2[0]), PAeq(nu2)), nu2_, f)
        us.append(sol[1:])
        alphas.append(sol[0])

        i += 1

    return alphas, us
