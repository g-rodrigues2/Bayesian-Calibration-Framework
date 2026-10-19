"""Tumor growth models and ODE solver dispatch.

Available models (suffix passed as `model_extension`):
  _exp, _mendel, _bert, _lin, _gomp, _surf, _log  -- classical one-dimensional growth laws

Each model_*(y, t, ...) is the right-hand side of the ODE; integration is done by
``solve_model()`` according to ``CALIB_SOLVER``:
  compiled   (default) exact solution of _exp and _log, compiled Dormand-Prince solver
             (compiled_ode.py) for the other laws
  analytic   exact solution of _exp and _log, odeint for the other laws
  odeint     scipy's odeint for every law
  solve_ivp  scipy's solve_ivp (CALIB_SOLVER_METHOD, CALIB_SOLVER_RTOL, CALIB_SOLVER_ATOL)

Layout of theta expected by _split_theta:
  [par1, par2, ..., T0_1, ..., T0_ndim, std]

``ndim`` is the number of free initial conditions, not the dimension of theta: the number of
mice of the group in the global mode and 1 in the mouse-specific mode. This is the only
difference between the two calibration modes (see inference.log_likelihood).
"""
import os

import numpy as np
from numba import jit
from scipy.integrate import odeint, solve_ivp

import compiled_ode


# === ODE solver configuration (read at import, inherited by spawned workers) ===
SOLVER        = os.environ.get('CALIB_SOLVER', 'compiled').lower()
SOLVER_METHOD = os.environ.get('CALIB_SOLVER_METHOD', 'LSODA')
SOLVER_RTOL   = float(os.environ.get('CALIB_SOLVER_RTOL', '1e-6'))
SOLVER_ATOL   = float(os.environ.get('CALIB_SOLVER_ATOL', '1e-8'))


# === One-dimensional models ===

@jit(nopython=True)
def model_exp(y, t, growth_rate):
    tumor = y
    return growth_rate * tumor


@jit(nopython=True)
def model_bert(y, t, growth_rate, b):
    tumor = y
    return growth_rate * (tumor ** (2./3.)) - b * tumor


@jit(nopython=True)
def model_mendel(y, t, growth_rate, power):
    tumor = y
    return growth_rate * (tumor ** power)


@jit(nopython=True)
def model_log(y, t, growth_rate, carrying_capacity):
    tumor = y
    return growth_rate * tumor * (1 - tumor / carrying_capacity)


@jit(nopython=True)
def model_lin(y, t, growth_rate, carrying_capacity):
    tumor = y
    return growth_rate * tumor / (tumor + carrying_capacity)


@jit(nopython=True)
def model_gomp(y, t, growth_rate, b, c):
    tumor = y
    return growth_rate * tumor * np.log(b / (tumor + c))


@jit(nopython=True)
def model_surf(y, t, growth_rate, carrying_capacity):
    tumor = y
    return growth_rate * tumor / ((tumor + carrying_capacity) ** (1./3.))


# Supported models (used by bounds.py and convergence_check.py)
ALL_MODELS = ['_exp', '_mendel', '_bert', '_lin', '_gomp', '_surf', '_log']


def _split_theta(theta, model_extension, ndim):
    """Splits theta into (ode_args, ic_for_mouse(idx)).

    theta = [par1..parN, T0_1..T0_ndim, std];  ic_for_mouse(idx) = T0[idx].
    """
    tail = ndim + 1  # initial conditions + std
    ode_args = tuple(theta[:-tail])
    T0s = theta[-tail:-1]

    def ic_fn(idx):
        return T0s[idx]

    return ode_args, ic_fn


# === Right-hand side: the compiled (numba) function, not .py_func ===
# .py_func discards the compilation and costs about 8x more per solve. The jitted function is probed
# once per model; if numba rejects it, the pure Python function is used (same result).
_RHS_CACHE = {}


def _rhs(model_extension):
    fn = globals()['model' + model_extension]
    cached = _RHS_CACHE.get(model_extension)
    if cached is not None:
        return cached
    py = getattr(fn, 'py_func', fn)
    try:
        import inspect as _i
        n = len(_i.signature(py).parameters) - 2      # y, t
        fn(np.array([100.0]), 0.0, *((0.5,) * n))
        chosen = fn
    except Exception:
        chosen = py
    _RHS_CACHE[model_extension] = chosen
    return chosen


# === Integration ===

def _closed_form(model_extension, t_array, y0, parameters):
    """Exact solutions of the exponential and logistic models (CALIB_SOLVER=analytic), shape (n, 1)."""
    t = np.asarray(t_array, dtype=float) - float(t_array[0])
    v0, rate = float(y0[0]), float(parameters[0])
    if model_extension == '_exp':
        return (v0 * np.exp(rate * t))[:, None]
    cc = float(parameters[1])
    with np.errstate(over='ignore'):
        return (cc * v0 / (v0 + (cc - v0) * np.exp(-rate * t)))[:, None]


CLOSED_FORM = ('_exp', '_log')


def _integrate(rhs, t_array, y0, parameters):
    """Dispatches to odeint (default, faster) or solve_ivp."""
    if SOLVER == 'solve_ivp':
        def f_adapter(t, y, *args):
            return rhs(y, t, *args)
        t_span = (float(t_array[0]), float(t_array[-1]))
        sol = solve_ivp(
            f_adapter, t_span, y0,
            t_eval=t_array, args=parameters,
            method=SOLVER_METHOD, rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
            dense_output=False,
        )
        if sol.status < 0:
            return np.full((len(t_array), len(y0)), np.nan)
        return sol.y.T
    return odeint(rhs, t=t_array, y0=y0, args=parameters, mxstep=2000)


def solve_model(model_extension, time_array, parameters, initial_condition,
                type_sol='data'):
    """Integrates the model identified by model_extension from time_array[0] to time_array[-1].
    type_sol='smooth' returns the solution on a 0.1-day grid (for plots)."""
    rhs = _rhs(model_extension)

    if np.isscalar(initial_condition):
        y0 = [float(initial_condition)]
    else:
        y0 = list(initial_condition)

    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p - bgn_p) / 0.1) + 1)

    if SOLVER in ('analytic', 'compiled') and model_extension in CLOSED_FORM:
        sol = _closed_form(model_extension, time_array, y0, parameters)
    elif SOLVER == 'compiled' and len(y0) == 1:
        sol = compiled_ode.solve(rhs, time_array, parameters, y0[0])
        if not np.all(np.isfinite(sol)):
            sol = _integrate(rhs, time_array, y0, parameters)     # stiff case: odeint
    else:
        sol = _integrate(rhs, time_array, y0, parameters)

    if type_sol == 'smooth':
        return np.column_stack((time_array, sol))
    return sol
