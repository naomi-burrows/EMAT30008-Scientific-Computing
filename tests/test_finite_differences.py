import numpy as np
import finite_difference
from math import pi
import unittest

def u_I(x, L):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y

def boundary(x, t):
    return 0

def u_exact(x, t, kappa, L):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
    return y

class TestFiniteDifferences(unittest.TestCase):

    def setUp(self):
        # Set problem parameters/functions
        self.kappa = 1.0  # diffusion constant
        self.L = 1.0  # length of spatial domain
        self.T = 0.5  # total time to solve for

        # Set numerical parameters
        self.mx = 10  # number of gridpoints in space
        self.mt = 1000  # number of gridpoints in time

    def test_FE(self):
        # Test forward Euler against exact solution
        x, u_j = finite_difference.finite_difference(lambda x: u_I(x, self.L), boundary, self.kappa, self.L, self.T, self.mx,
                                                     self.mt, method='FE')
        self.assertAlmostEqual(u_j.all(), u_exact(x, self.T, self.kappa, self.L).all())

    def test_BE(self):
        # Test backward Euler against exact solution
        x, u_j = finite_difference.finite_difference(lambda x: u_I(x, self.L), boundary, self.kappa, self.L, self.T, self.mx,
                                                     self.mt, method='BE')
        self.assertAlmostEqual(u_j.all(), u_exact(x, self.T, self.kappa, self.L).all())

    def test_CN(self):
        # Test Crank Nicholson against exact solution
        x, u_j = finite_difference.finite_difference(lambda x: u_I(x, self.L), boundary, self.kappa, self.L, self.T, self.mx,
                                                     self.mt, method='CN')
        self.assertAlmostEqual(u_j.all(), u_exact(x, self.T, self.kappa, self.L).all())

if __name__ == '__main__':
    unittest.main()