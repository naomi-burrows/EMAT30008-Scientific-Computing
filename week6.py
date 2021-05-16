import numpy as np
import matplotlib.pyplot as plt
from numerical_continuation import natural_parameter_continuation

def cubic_eq(x, c):
    return x**3 -x +c

# Solve by natural parameter continuation
cs, xs = natural_parameter_continuation(cubic_eq, 1.5, (2, -2), 0.001)

# Plot results - x against c
plt.plot(cs, xs)
plt.show()

# Plot results - f(x) against c, should all be zero
plt.plot(cs, cubic_eq(np.array(xs), np.array(cs)))
plt.show()