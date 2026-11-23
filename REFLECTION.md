# Project Reflection: Numerical Solutions of ODEs

- **Course:** Scientific Programming for Physics (Oct 27, 2026)
- **Author:** Durdona Uskenbayeva
- **Advisor:** Aleksandr Kapitonov
- **Repository:** `scientific-programming-ode`

## 1. Summary of Completed Work
In this project, we implemented and analyzed numerical methods for ordinary differential equations using `USKENBAYEVA_DURDONA_260620.ipynb`, `lecture_executed.html`, modular solver code in `src/ode_solvers.py`, and automated tests in `tests/test_ode_solvers.py`:
- **Generalized Euler Method (`euler_method`):** Studied first-order accuracy on `dy/dt = -2y` and observed numerical instability when the step size is too large (`h = 1.1`).
- **Harmonic Oscillator & Energy Conservation:** Solved `y'' = -y` with `RK45` and checked total mechanical energy `E = 0.5 * (v^2 + y^2)`.
- **Stiff Solvers (`RK45`, `Radau`, `BDF`):** Compared function evaluations (`nfev`) on `dy/dt = -1000y`.
- **Van der Pol Oscillator (`vanderpol`):** Simulated phase trajectories starting inside and outside the limit cycle (`mu = 1.5`).

## 2. Key Lessons
- Choosing a step size beyond the stability limit causes explicit Euler solutions to diverge.
- Implicit solvers (`Radau`, `BDF`) handle stiff equations efficiently compared to explicit solvers on stiff test cases.
- Organizing reusable functions into `src/` and verifying them with `pytest` improves code reliability and reproducibility.
