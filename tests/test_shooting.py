import numpy as np
import shooting
import unittest

def hopf(u, t, beta, sigma=-1.0):
    """
    ODE for the Hopf bifurcation normal form. Returns du1/dt and du2/dt

        Parameters:
            u (tuple):      vector in the form (u1, u2)
            t (float):      value for time
            beta (float):   parameter in equations
            sigma (float):  default=-1.0. Parameter in equations

        Returns:
            numpy array of (du1/dt, du2/dt)
    """
    u1, u2 = u

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))

def real_sol(t, beta, theta=0.0):
    """
    Analytic solution of the Hopf bifurcation normal form. Returns u1 and u2

        Parameters:
            t (float):      value for time
            beta (float):   parameter in equations
            theta (float):  default=0. Phase

        Returns:
            numpy array of (u1, u2)
    """
    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)
    return np.array((u1, u2))


class TestShooting(unittest.TestCase):

    def setUp(self):
        self.t = np.linspace(0, 40, 401)
        self.u0, self.T = shooting.find_limit_cycle(lambda u, t: hopf(u, t, beta=1), (-1, 0), 6)

    def test_T(self):
        self.assertAlmostEqual(self.T, np.pi*2)

    def test_u0(self):
        self.assertAlmostEqual(self.u0[0], -1)
        self.assertAlmostEqual(self.u0[1], 0)

if __name__ == '__main__':
    unittest.main()