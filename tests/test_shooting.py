import numpy as np
import shooting
import ode_solver
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
        # Find limit cycle with numerical shooting
        t = np.linspace(0, 40, 401)
        self.u0, self.T = shooting.find_limit_cycle(lambda u, t: hopf(u, t, beta=1), (-1, 0), 6)

        # Find solution values using solve_ode from ode_solver
        t = np.linspace(0, self.T, 101)
        sol = ode_solver.solve_ode(lambda u, t: hopf(u, t, beta=1), self.u0, t, ode_solver.rk4_step, 0.001)
        self.u1 = sol[:, 0]
        self.u2 = sol[:, 1]

        # Find solution values using analytic solution for same t range
        self.u1_real, self.u2_real = real_sol(t, beta=1, theta=np.pi)

    def test_T(self):
        # Test that time period found is equal to 2pi
        self.assertAlmostEqual(self.T, np.pi*2)

    def test_u0(self):
        # Test that initial conditions found are the same as the real solution start point
        self.assertAlmostEqual(self.u0[0], self.u1_real[0])
        self.assertAlmostEqual(self.u0[1], self.u2_real[0])

    def test_endpoints(self):
        # Test that the end point = start point
        self.assertAlmostEqual(self.u1[-1], self.u1[0])
        self.assertAlmostEqual(self.u2[-1], self.u2[0])

    def test_allpoints(self):
        # Test that all points calculated match the analytic solution
        self.assertAlmostEqual(self.u1.all(), self.u1_real.all())
        self.assertAlmostEqual(self.u2.all(), self.u2_real.all())

if __name__ == '__main__':
    unittest.main()