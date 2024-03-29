# Finite difference solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with given boundary conditions and initial temperature distribution

import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from math import pi

def finite_difference(u_I, boundary, kappa, L, T, mx, mt, method):
    """
    Solves PDE using finite differences.

        Parameters:
            u_I (function):         Initial temperature distribution as a function of x
            boundary (function):    Boundary condition, function of x and t.
            kappa (float):          Diffusion constant
            L (float):              Length of spatial domain
            T (float):              Total time to solve for
            mx (int):               Number of gridpoints in space
            mt (int):               Number of gridpoints in time
            method (str):           The method to use. 'FE' for forward Euler, 'BE' for backward Euler, 'CN' for Crank Nicholson.

        Returns:
            x, u_j (the values of u at each x at time T)

    """

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
    print("deltax=", deltax)
    print("deltat=", deltat)
    print("lambda=", lmbda)

    # Set up matrix/matrices
    if method == 'FE':
        diagonals = [[1 - 2*lmbda] * mx, [lmbda] * (mx-1), [lmbda] * (mx-1)]
        A_FE = scipy.sparse.diags(diagonals, [0, -1, 1], format = 'csc')
    elif method == 'BE':
        diagonals = [[1 + 2 * lmbda] * mx, [- lmbda] * (mx - 1), [- lmbda] * (mx - 1)]
        A_BE = scipy.sparse.diags(diagonals, [0, -1, 1], format = 'csc')
    elif method == 'CN':
        diagonals = [[1 + lmbda] * mx, [-lmbda / 2] * (mx - 1), [-lmbda / 2] * (mx - 1)]
        A_CN = scipy.sparse.diags(diagonals, [0, -1, 1], format = 'csc')
        diagonals = [[1 - lmbda] * mx, [lmbda / 2] * (mx - 1), [lmbda / 2] * (mx - 1)]
        B_CN = scipy.sparse.diags(diagonals, [0, -1, 1], format = 'csc')
    else:
        print('Choose a method:\n   \'FE\' - forward Euler\n   \'BE\' - backward Euler\n   \'CN\' - Crank Nicholson')

    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i])

    # Solve the PDE: loop over all time points
    for j in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]

        if method == 'FE':
            u_jp1[1:] = A_FE.dot(u_j[1:])
        elif method == 'BE':
            u_jp1[1:] = scipy.sparse.linalg.spsolve(A_BE, u_j[1:])
        elif method == 'CN':
            u_jp1[1:] = scipy.sparse.linalg.spsolve(A_CN, B_CN.dot(u_j[1:]))

        # Boundary conditions
        u_jp1[0] = boundary(0, t[j])
        u_jp1[mx] = boundary(L, t[j])

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return x, u_j

if __name__ == "__main__":

    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant
    L = 1.0  # length of spatial domain
    T = 0.5  # total time to solve for

    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y

    def boundary(x, t):
        return 0

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    # Set numerical parameters
    mx = 10  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time

    # Solve using each method and plot results
    # Forward Euler
    x, u_j = finite_difference(u_I, boundary, kappa, L, T, mx, mt, method='FE')
    plt.scatter(x, u_j)
    # Backward Euler
    x, u_j = finite_difference(u_I, boundary, kappa, L, T, mx, mt, method='BE')
    plt.scatter(x, u_j)
    # Crank Nicholson
    x, u_j = finite_difference(u_I, boundary, kappa, L, T, mx, mt, method='CN')
    plt.scatter(x, u_j)

    # Plot exact solution
    xx = np.linspace(0, L, 250)
    plt.plot(xx, u_exact(xx, T), color='red')
    plt.xlabel('x')
    plt.ylabel('u(x,'+str(T)+')')
    plt.legend(['Exact', 'Forward Euler', 'Backward Euler', 'Crank Nicholson'])
    plt.show()