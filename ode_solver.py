import numpy as np

def euler_step(f, x1, t1, h):
    # function does a single Euler step
    x2 = x1 + h* f(x1,t1)
    t2 = t1 + h
    return [x2, t2]

def solve_to(step, f, x1, t1, t2, hmax):
    # function solves from x1,t1 to x2,t2 in steps no bigger than hmax
    num_steps = int((t2 - t1) / hmax)
    x = x1
    t = t1
    for i in range(num_steps):
        x,t = step(f, x, t, hmax)
    if t != t2:
        h = t2 - t
        x,t = step(f, x, t, h)
    return x

def solve_ode(f, x0, t, method, hmax):
    # function returns a series of numerical solution estimates to ode f
    x = np.zeros_like(t)
    x[0] = x0[0]
    for i in range(1, len(t)):
        x[i] = solve_to(method, f, x[i-1], t[i-1], t[i], hmax)
    return x