import numpy as np
import numerical_continuation
import unittest

def cubic_eq(x, c):
    return x**3 -x +c

class TestContinuation(unittest.TestCase):

    def test_NP(self):
        # Test that the cubic equation can be solved by natural parameter continuation by checking that f(x) = 0 for all c
        cs, xs = numerical_continuation.natural_parameter_continuation(cubic_eq, 1.5, (-2, 2), 0.001,
                                                discretisation=lambda u0_, f, alpha: f(u0_, alpha))
        self.assertAlmostEqual(cubic_eq(np.array([x[0] for x in xs]), np.array(cs)).all(), 0)

    def test_PA(self):
        # Test that the cubic equation can be solved by pseudo-arclength continuation by checking that f(x) = 0 for all c
        cs, xs = numerical_continuation.pseudo_arclength(cubic_eq, 1.5, (-2, 2), 0.001,
                                                discretisation=lambda u0_, f, alpha: f(u0_, alpha))
        self.assertAlmostEqual(cubic_eq(np.array([x[0] for x in xs]), np.array(cs)).all(), 0)

if __name__ == '__main__':
    unittest.main()