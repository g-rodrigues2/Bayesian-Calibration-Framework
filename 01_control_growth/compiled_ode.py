"""Compiled solver of the one-dimensional tumor models (default solver of stages 01, 03 and 05).

The right-hand sides of the models are numba functions with the odeint signature,

    growth models     model(y, t, *parameters)
    treatment models  model(y, t, r, cc, *treatment parameters, tau, dose_schedule)

and the treatment effects switch on at the administration days (tau_i, or tau_i + delay for the delayed
radiotherapy models), where the right-hand side jumps. `solve` integrates a model in nopython mode with the
explicit Dormand-Prince 5(4) pair and adaptive steps, restarting at every switching time and stopping
exactly at every requested time, so that no step crosses a discontinuity and no interpolation is needed.
Stages that fall on the end of a segment are evaluated 1e-10 day before it, on the side of the jump that
the segment belongs to.

With the default tolerances (rtol 1e-9, atol 1e-8) the volumes agree with a DOP853 reference (rtol 1e-12)
restarted at every administration to a relative error below 1e-5 over the whole prior boxes of stages 3 and 5,
including effects that last a small fraction of a day (tests/test_compiled_ode.py); one solve is 10 to 50 times
faster than odeint. Every segment starts with a small step (H_RESTART), so a narrow effect pulse that starts
at an administration is resolved; scipy's odeint, which does not restart, can step over such pulses (for
example the accelerated cumulative-dose immunotherapy term with a large decay rate), and then overestimates
the tumor volume. When the explicit steps are exhausted (stiff effects, e.g. very fast decays of
the accelerated immunotherapy terms) `solve` returns NaN and the callers integrate with odeint instead, so no
parameter vector that odeint can integrate is rejected. CALIB_SOLVER=odeint restores scipy's odeint in every stage.
"""
import inspect
import os

import numpy as np
from numba import njit

RTOL = 1e-9
ATOL = 1e-8
MAX_STEPS = 50_000       # beyond this the problem is stiff and the callers fall back to odeint
_EDGE = 1e-10
H_RESTART = 1e-3

# Dormand-Prince 5(4) coefficients
A21 = 1.0 / 5.0
A31, A32 = 3.0 / 40.0, 9.0 / 40.0
A41, A42, A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
A51, A52, A53, A54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
A61, A62, A63, A64, A65 = 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0
B1, B3, B4, B5, B6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0
E1, E3, E4, E5, E6, E7 = (71.0 / 57600.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0,
                          -1.0 / 40.0)
C2, C3, C4, C5 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0


def use_compiled():
    return os.environ.get('CALIB_SOLVER', 'compiled').lower() == 'compiled'


def _takes_scalar(py, n_args, treatment):
    """True if the pure Python right-hand side accepts a float volume (else it indexes y[0])."""
    args = [0.5] * n_args
    if treatment:
        args = [0.1, 1000.0] + [0.1] * (n_args - 4) + [np.array([0.0]), np.array([1.0])]
    try:
        out = py(100.0, 0.0, *args)
        return np.ndim(out) == 0
    except Exception:
        return False


def _wrap(fn, treatment):
    """Compiled rhs(t, y, p, tau, dose) -> float of a model with 0 to 6 scalar parameters."""
    py = getattr(fn, 'py_func', fn)
    n_args = len(inspect.signature(py).parameters) - 2
    n_scalar = n_args - 2 if treatment else n_args
    scalar = _takes_scalar(py, n_args, treatment)
    src = ['def rhs(t, y, p, tau, dose):']
    yarg = 'y' if scalar else 'np.array([y])'
    pars = ', '.join(f'p[{i}]' for i in range(n_scalar))
    call = f'fn({yarg}, t' + (f', {pars}' if pars else '') + (', tau, dose' if treatment else '') + ')'
    src.append(f'    return {call}' if scalar else f'    v = {call}\n    return v')
    namespace = {'fn': fn, 'np': np}
    exec('\n'.join(src), namespace)
    return njit(namespace['rhs'])


def _make_solver(rhs):
    @njit
    def segment(t, y, t_end, h, p, tau, dose, rtol, atol, budget):
        steps = 0
        k1 = rhs(t, y, p, tau, dose)
        edge = t_end - _EDGE
        while t < t_end:
            if steps >= budget:
                return np.nan, h, steps
            last = t + h >= t_end
            h_try = t_end - t if last else h
            k2 = rhs(min(t + C2 * h_try, edge), y + h_try * A21 * k1, p, tau, dose)
            k3 = rhs(min(t + C3 * h_try, edge), y + h_try * (A31 * k1 + A32 * k2), p, tau, dose)
            k4 = rhs(min(t + C4 * h_try, edge), y + h_try * (A41 * k1 + A42 * k2 + A43 * k3), p, tau, dose)
            k5 = rhs(min(t + C5 * h_try, edge), y + h_try * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4),
                     p, tau, dose)
            k6 = rhs(min(t + h_try, edge),
                     y + h_try * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5), p, tau, dose)
            y_new = y + h_try * (B1 * k1 + B3 * k3 + B4 * k4 + B5 * k5 + B6 * k6)
            t_new = t_end if last else t + h_try
            k7 = rhs(min(t_new, edge), y_new, p, tau, dose)
            err = h_try * (E1 * k1 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7)
            ratio = abs(err) / (atol + rtol * max(abs(y), abs(y_new)))
            steps += 1
            if not np.isfinite(ratio):
                h = 0.2 * h_try
                if h < 1e-14:
                    return np.nan, h, steps
                continue
            if ratio <= 1.0:
                t = t_new
                y = y_new
                k1 = k7
                factor = 10.0 if ratio == 0.0 else min(10.0, 0.9 * ratio ** -0.2)
                if not last or h_try >= h:
                    h = h_try * factor
            else:
                h = h_try * max(0.2, 0.9 * ratio ** -0.2)
                if h < 1e-14:
                    return np.nan, h, steps
        return y, h, steps

    @njit
    def solve(y0, times, p, tau, dose, switches, rtol, atol):
        n = times.shape[0]
        out = np.empty(n)
        t = times[0]
        y = y0
        out[0] = y0
        events = np.sort(np.concatenate((switches, times[1:])))
        k = 1
        while k < n and times[k] <= t:
            out[k] = y
            k += 1
        h = 0.1
        budget = MAX_STEPS
        for e in range(events.shape[0]):
            t_end = events[e]
            if t_end <= t:
                continue
            if k >= n:
                break
            # restart with a small step: an effect that starts at t may be a pulse much narrower than the last step
            y, h, used = segment(t, y, t_end, min(h, H_RESTART), p, tau, dose, rtol, atol, budget)
            budget -= used
            if not np.isfinite(y):
                out[k:] = np.nan
                return out
            t = t_end
            while k < n and times[k] <= t:
                out[k] = y
                k += 1
        return out

    return solve


_SOLVERS = {}
_EMPTY = np.zeros(0)


def solve(fn, times, parameters, y0, tau=None, dose_schedule=None, delay=0.0, rtol=RTOL, atol=ATOL):
    """Volumes at `times` (times[0] is the initial time), shape (n, 1) as odeint.

    Growth models: tau is None. Treatment models: parameters = (r, cc, *treatment parameters); the
    effects switch on at tau + delay."""
    treatment = tau is not None
    key = (fn, treatment)
    if key not in _SOLVERS:
        _SOLVERS[key] = _make_solver(_wrap(fn, treatment))
    times = np.ascontiguousarray(times, dtype=float)
    if treatment:
        tau = np.asarray(tau, dtype=float)
        dose_schedule = np.asarray(dose_schedule, dtype=float)
        switches = tau + delay
        switches = switches[(switches > times[0]) & (switches < times[-1])]
    else:
        tau = dose_schedule = switches = _EMPTY
    p = np.asarray(parameters, dtype=float)
    out = _SOLVERS[key](float(y0), times, p, tau, dose_schedule, switches, rtol, atol)
    return out.reshape(-1, 1)
