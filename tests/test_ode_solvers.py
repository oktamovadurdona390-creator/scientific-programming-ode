"""
Automated test suite for `USKENBAYEVA_DURDONA_260620.ipynb` and `src/ode_solvers.py`.
Run with: pytest -v
"""

import numpy as np
from scipy.integrate import solve_ivp

from src.ode_solvers import (
    euler_method,
    dydt,
    harmonic_system,
    stiff_system,
    vanderpol,
)


def test_euler_shape_and_initial_condition():
    """Verify Euler method output shape, length, and initial condition."""
    t_test_e, y_test_e = euler_method(dydt, 0.0, 1.0, 0.1, 1.0)
    assert len(t_test_e) == len(y_test_e), "Euler output arrays must have the same length!"
    assert y_test_e[0, 0] == 1.0, "Initial condition for Euler method is incorrect!"
    assert y_test_e.ndim == 2 and y_test_e.shape[1] == 1


def test_euler_convergence_rate():
    """Verify first-order O(h) error reduction on dy/dt = -2y."""
    t0, y0, t_end = 0.0, 1.0, 5.0
    t_coarse, y_coarse = euler_method(dydt, t0, y0, 0.1, t_end)
    t_fine, y_fine = euler_method(dydt, t0, y0, 0.05, t_end)

    err_coarse = np.max(np.abs(y_coarse[:, 0] - np.exp(-2.0 * t_coarse)))
    err_fine = np.max(np.abs(y_fine[:, 0] - np.exp(-2.0 * t_fine)))
    assert err_fine < err_coarse * 0.6


def test_rk45_analytic_accuracy():
    """Verify SciPy RK45 solver matches exact analytic solution y(t) = exp(-2t)."""
    sol_test = solve_ivp(dydt, [0.0, 1.0], [1.0], t_eval=[1.0])
    expected_val = np.exp(-2.0 * 1.0)
    np.testing.assert_allclose(
        sol_test.y[0][0],
        expected_val,
        rtol=1e-2,
        err_msg="RK45 solution deviates too much from the exact value!",
    )


def test_harmonic_oscillator_energy_conservation():
    """Verify Harmonic Oscillator shape and energy conservation."""
    t0_h, t_end_h, h_h = 0.0, 20.0, 0.05
    y0_h = np.array([1.0, 0.0])
    t_eval_h = np.arange(t0_h, t_end_h + h_h, h_h)
    sol_h = solve_ivp(harmonic_system, [t0_h, t_end_h], y0_h, t_eval=t_eval_h, method="RK45")

    assert sol_h.y.shape[0] == 2, "Harmonic oscillator must have 2 state variables!"
    energy = 0.5 * (sol_h.y[1] ** 2 + sol_h.y[0] ** 2)
    energy_drift = np.max(np.abs(energy - energy[0]))
    assert energy_drift < 0.01, "Energy conservation check failed: energy drift is too high!"


def test_stiff_solvers_efficiency_and_agreement():
    """Verify implicit stiff solvers (Radau, BDF) require fewer evaluations than RK45."""
    t_span = [0.0, 1.0]
    y0 = np.array([1.0])
    t_eval = np.linspace(0.0, 1.0, 100)

    sol_rk = solve_ivp(stiff_system, t_span, y0, method="RK45", t_eval=t_eval)
    sol_radau = solve_ivp(stiff_system, t_span, y0, method="Radau", t_eval=t_eval)
    sol_bdf = solve_ivp(stiff_system, t_span, y0, method="BDF", t_eval=t_eval)

    assert sol_radau.nfev < sol_rk.nfev
    assert sol_bdf.nfev < sol_rk.nfev
    np.testing.assert_allclose(sol_radau.y[0], sol_bdf.y[0], atol=1e-4)


def test_vanderpol_limit_cycle_convergence():
    """Verify trajectories converge toward the Van der Pol limit cycle."""
    t_span = [0.0, 25.0]
    sol_in = solve_ivp(lambda t, y: vanderpol(t, y, mu=1.5), t_span, [0.2, 0.2], t_eval=[25.0])
    sol_out = solve_ivp(lambda t, y: vanderpol(t, y, mu=1.5), t_span, [3.0, 2.0], t_eval=[25.0])

    r_in = np.hypot(sol_in.y[0, -1], sol_in.y[1, -1])
    r_out = np.hypot(sol_out.y[0, -1], sol_out.y[1, -1])
    assert r_in > 1.5 and r_out > 1.5
