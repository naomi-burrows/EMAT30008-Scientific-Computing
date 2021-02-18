def euler_step(f, x1, t1, h):
    # function does a single Euler step
    x2 = x1 + h* f(x1,t1)
    t2 = t1 + h
    return [x2, t2]