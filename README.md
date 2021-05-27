# EMAT30008
EMAT30008 is a collection of python files which perform various numerical methods to solve ODEs and PDEs.

#### Solving ODEs
* To solve an ODE using forward Euler or 4th-order Runge-Kutta (RK4), use `ode_solver.solve_ode` with method argument:
    * `ode_solver.euler_step` for forward Euler
    * `ode_solver.rk4_step` for RK4

#### Numerical Shooting
* To find limit cycles in an ODE, use `shooting.find_limit_cycle`
* For the general shooting discretisation, use `shooting.shoot`

#### Numerical Continuation
* For natural parameter continuation, use `numerical_continuation.natural_parameter_continuation`
* For pseudo-arclength continuation, use `numerical_continuation.pseudo_arclength`

#### Finite Difference
* To solve a PDE in the form of the 1D heat equation using finite differences, use `finite_difference.finite_difference` with method argument:
    * `'FE'` for forward Euler
    * `'BE'` for backward Euler
    * `'CN'` for Crank Nicholson