import numpy as np
from scipy.optimize import fsolve
import ode_solver

# G(U0) = u0 - f(u0, T)
def G(f, u0, t0, T):
    # Function solves the ODE and returns the difference between u0 and the solution
    sol = ode_solver.solve_ode(f, u0, [t0, T], ode_solver.rk4_step, 0.01)
    return u0 - sol[-1]

# phase condition is of form d?/dt=0 (variable represented by ? is determined by the phase_index)
def phase_condition(f, u0, phase_index):
    return np.array([f(u0, 0)[phase_index]]) # return dy/dt at u0

def shoot(U, f, phase_index):
    # U is in the form of initial guesses for variables followed by T
    u0 = U[:-1]
    T = U[-1]
    return np.concatenate((G(f, u0, 0, T), phase_condition(f, u0, phase_index)))

def find_limit_cycle(f, u0_, T_, phase_index=0):
    sol = fsolve(lambda U, f: shoot(U, f, phase_index), np.concatenate((u0_, [T_])), f)
    u0 = sol[:-1]
    T = sol[-1]
    return u0, T
