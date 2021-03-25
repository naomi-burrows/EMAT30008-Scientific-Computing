import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from math import nan
import ode_solver

def plot_time_series(ode, u0, t, labels=[], show=True):
    """
    Plots the time series of provided ode for provided time interval.

        Parameters:
            ode (function): ode to plot the time series of
            u0 (tuple):     initial conditions
            t (ndarray):    values of t to plot
            labels (list):  optional,   1 value   -> y-axis label
                                        2+ values -> legend
            show (bool):    optional, default = True.
                                        True  -> shows graph
                                        False -> does not show graph (allows multiple plots on the same graph)
        Returns:
            None
    """

    # solve ode
    sol = ode_solver.solve_ode(ode, u0, t, ode_solver.rk4_step, 0.001)

    # unpack solution
    for i in range(len(u0)):
        var = sol[:, i]
        plt.plot(t, var)

    plt.xlabel('time, t')

    # if only one label give, use as y-axis label, otherwise use as legend.
    if len(labels) == 1:
        plt.ylabel(labels[0])
    else:
        plt.legend(labels)

    if show:
        plt.show()

def plot_phase_portrait(ode, u0, t, labels=['x', 'y'], show=True):
    """
    Plots the phase portrait of provided ode for provided time interval.

        Parameters:
            ode (function): ode to plot the phase portrait of
            u0 (tuple):     initial conditions
            t (ndarray):    values of t to plot
            labels (list):  optional, length = 2, default = ['x', 'y']. These are the axis labels.
            show (bool):    optional, default = True.
                                        True  -> shows graph
                                        False -> does not show graph (allows multiple plots on the same graph)
        Returns:
            None
    """

    # solve ode
    sol = ode_solver.solve_ode(ode, u0, t, ode_solver.rk4_step, 0.001)

    # unpack solution
    x = sol[:, 0]
    y = sol[:, 1]

    plt.plot(x, y)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if show:
        plt.show()

def plot_nullcline(ode, x0, xrange, index=0, points=101, show=True):
    """
    Plots the nullcline for variable indicated by index.

        Parameters:
            ode (function): ode to plot nullclines of
            x0 (float):     initial condition for nullcline variable
            xrange (tuple): range to plot nullcline for
            index (int):    optional, default=0. This is the variable for which the nullcline is plotted
            xpoints (int):  number of points to plot
            show (bool):    optional, default = True.
                                        True  -> shows graph
                                        False -> does not show graph (allows multiple plots on the same graph)
        Returns:
            None
    """
    xval = np.linspace(min(xrange), max(xrange), points)
    yval = np.zeros(np.size(xval))

    for (i, x) in enumerate(xval):
        result = root(lambda N: ode((x, N), nan)[index], x0)
        if result.success:
            yval[i] = result.x
        else:
            yval[i] = nan

    plt.plot(xval, yval)

    if show:
        plt.show()