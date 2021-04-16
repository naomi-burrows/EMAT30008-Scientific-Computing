import numpy as np
from scipy.optimize import fsolve
import ode_solver

def G(f, u0, t0, T):
    """
    Solves the ODE and returns the difference between u0 and the solution.

        Parameters:
            f (function):   the ode to be solved with numerical shooting
            u0 (tuple):     estimate of initial conditions
            t0 (float):     initial time
            T (tuple):      estimate of time period

        Returns:
            the difference between the solution found and the initial estimate, u0
    """
    sol = ode_solver.solve_ode(f, u0, [t0, T], ode_solver.rk4_step, 0.01)
    return u0 - sol[-1]

# phase condition is of form d?/dt=0 (variable represented by ? is determined by the phase_index)
def phase_condition(f, u0, phase_index):
    """
    The phase condition to be used for numerical shooting.
    It is of the form d?/dt=0 (where the variable represented by ? is determined by the phase_index).

        Parameters:
            f (function):       the ode to be solved with numerical shooting
            u0 (tuple):         initial guess
            phase_index (int):  the index for the variable to be used. The phase condition is that d/dt(this variable) = 0

        Returns:
            the value of d/dt(variable determined by phase_index) at u0
    """
    return np.array([f(u0, 0)[phase_index]])

def shoot(U, f, phase_index):
    """
    Function returns the array to be solved for roots in find_limit_cycle.

        Parameters:
            U (tuple):          initial guesses for variables and time period
            f (function):       the ode to be solved with numerical shooting
            phase_index (int):  the index for the variable to be used for the phase condition. d/dt(this variable) = 0

        Returns:
            an array of u0-solution and the phase condition. This will make up a system of equations to be solved.
    """
    u0 = U[:-1]
    T = U[-1]
    return np.concatenate((G(f, u0, 0, T), phase_condition(f, u0, phase_index)))

def find_limit_cycle(f, u0_, T_, phase_index=0):
    """
    Solves for u0 and T which make a limit cycle.

        Parameters:
            f (function):       the ode to find the limit cycle of
            u0_ (tuple):        an estimate for the initial conditions of the variables in f
            T_ (float):         an estimate for the time period
            phase_index (int):  optional, default=0. The index for the variable to be used for the phase condition. d/dt(this variable) = 0

        Returns:
            the solutions for u0 and T - the initial conditions and time period of the limit cycle
    """
    sol = fsolve(lambda U, f: shoot(U, f, phase_index), np.concatenate((u0_, [T_])), f)
    u0 = sol[:-1]
    T = sol[-1]
    return u0, T
