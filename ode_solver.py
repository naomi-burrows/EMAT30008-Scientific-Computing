import numpy as np
import math

def euler_step(f, x1, t1, h):
    """
    Does a single Euler step. Returns [x_{n+1}, t_{n+1}].

        Parameters:
            f (function):   function to do the euler step on
            x1:             x_n - initial value(s) for x
            t1 (float):     t_n - inital time
            h (float):      stepsize

        Returns:
            [x_{n+1}, t_{n+1}] = [x at t + h, t + h]
    """
    x2 = x1 + h* np.array(f(x1,t1))
    t2 = t1 + h

    return [x2, t2]

def rk4_step(f, x1, t1, h):
    """
    Does a single RK4 step. Returns [x_{n+1}, t_{n+1}].

        Parameters:
            f (function):   function to do the RK4 step on
            x1:             x_n - initial value(s) for x
            t1 (float):     t_n - inital time
            h (float):      stepsize

        Returns:
            [x_{n+1}, t_{n+1}] = [x at t + h, t + h]
    """
    k1 = np.array(f(x1, t1))
    k2 = np.array(f(x1 + h*k1/2, t1 + h/2))
    k3 = np.array(f(x1 + h*k2/2, t1 + h/2))
    k4 = np.array(f(x1 + h*k3, t1 + h))

    x2 = x1 + h*(k1 + 2*k2 + 2*k3 + k4)/6
    t2 = t1 + h

    return [x2, t2]

def solve_to(step, f, x1, t1, t2, hmax):
    """
    Solves from x1,t1 to x2,t2 in steps no bigger than hmax. Returns x value at time t2.

        Parameters:
            step (function):    type of step to do (euler_step or rk4_step)
            f (function):       function to solve
            x1:                 x_n - initial value(s) for x
            t1 (float):         t_n - inital time
            t2 (float):         t_{n+1} - time to solve to
            hmax (float):       maximum stepsize

        Returns:
            solution(s) for x at time t2
    """
    num_steps = math.floor((t2 - t1) / hmax)
    x = x1
    t = t1
    for i in range(num_steps):
        x,t = step(f, x, t, hmax)
    if t != t2:
        h = t2 - t
        x,t = step(f, x, t, h)
    return x

def solve_ode(f, x0, t, method, hmax):
    """
    Returns a series of numerical solution estimates to ode f.

        Parameters:
            f (function):       function to solve
            x0:                 initial value(s) for x
            t (ndarray):        time values to approximate x for
            method (function):  method by which to approximate solution (euler_step or rk4_step)
            hmax (float):       maximum stepsize

        Returns:
            solutions for x at each time value in t
    """
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = solve_to(method, f, x[i-1], t[i-1], t[i], hmax)
    return x