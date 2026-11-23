"""
Numerical ODE Solvers and Physical Systems
Course: Scientific Programming for Physics (Oct 27, 2026)
Author: Durdona Uskenbayeva
Advisor: Aleksandr Kapitonov
"""

import numpy as np
from scipy.integrate import solve_ivp


def euler_method(f, t0, y0, h, t_end):
    """Generalized Explicit Euler Method (supports both scalar and vector systems)."""
    if h <= 0:
        raise ValueError("Step size h must be strictly positive.")
    t_values = np.arange(t0, t_end + h, h)
    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    y_values = np.zeros((len(t_values), len(y0_arr)), dtype=float)
    y_values[0] = y0_arr

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        y_values[i] = y + h * np.asarray(f(t, y), dtype=float)

    return t_values, y_values


def dydt(t, y):
    """Exponential decay governing ODE: dy/dt = -2y."""
    return -2.0 * np.asarray(y, dtype=float)


def harmonic_system(t, y):
    """Simple harmonic oscillator system: dy1/dt = y2, dy2/dt = -y1."""
    return np.array([y[1], -y[0]], dtype=float)


def stiff_system(t, y):
    """Stiff linear test problem: dy/dt = -1000y."""
    return np.array([-1000.0 * y[0]], dtype=float)


def vanderpol(t, y, mu=1.5):
    """Nonlinear Van der Pol oscillator: y'' - mu*(1 - y^2)*y' + y = 0."""
    return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]], dtype=float)


def solve_ode(f, t_span, y0, method="RK45", t_eval=None, rtol=1e-6, atol=1e-9):
    """Wrapper around scipy.integrate.solve_ivp for RK45, Radau, and BDF."""
    sol = solve_ivp(f, t_span, y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Solver {method} failed: {sol.message}")
    return sol
