import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import fsolve
from math import nan
import ode_solver

# G(U0) = u0 - f(u0, T)
def G(f, u0, t0, T):
    # Function solves the ODE and returns the difference between u0 and the solution
    sol = ode_solver.solve_ode(f, u0, [t0, T], ode_solver.rk4_step, 0.01)
    return u0 - sol[-1]

# phase condition is dy/dt=0
def phase_condition(f, u0):
    return np.array([f(u0, 0)[1]]) # return dy/dt at u0

def shoot(U, f):
    # U is in the form of initial guesses for variables followed by T
    u0 = U[:-1]
    T = U[-1]
    return np.concatenate((G(f, u0, 0, T), phase_condition(f, u0)))

def find_limit_cycle(f, u0_, T_):
    sol = fsolve(shoot, np.concatenate((u0_, [T_])), f)
    u0 = sol[:-1]
    T = sol[-1]
    return u0, T
