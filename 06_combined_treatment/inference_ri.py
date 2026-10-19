"""Gaussian likelihood with a residual standard deviation shared by all observations.

CALIB_RI_SOLVER selects the integrator: 'compiled' (default, rad_immuno_fast) or 'odeint'
(rad_immuno_models.solve_mouse, used by the original calibrations).
"""
import os

import numpy as np

import rad_immuno_fast
from rad_immuno_models import MODEL_PARAMS, solve_mouse

SOLVER = os.environ.get('CALIB_RI_SOLVER', 'compiled').lower()


def split_theta(theta, model, n_mice):
    k = len(MODEL_PARAMS[model])
    return tuple(theta[:k]), theta[k:k + n_mice], float(theta[-1])


def gaussian_loglik(obs, pred, std):
    return float(-0.5 * np.sum((pred - obs) ** 2 / std ** 2 + np.log(2 * np.pi * std ** 2)))


def log_likelihood(theta, model, mice, upstream):
    params, ics, std = split_theta(theta, model, len(mice))
    if not std > 0:
        return -np.inf
    if SOLVER == 'compiled':
        return rad_immuno_fast.log_likelihood(model, params, ics, std, mice, upstream)
    ll = 0.0
    for mouse, ic in zip(mice, ics):
        try:
            pred = solve_mouse(model, params, ic, mouse, upstream)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return -np.inf
        if not np.all(np.isfinite(pred)):
            return -np.inf
        ll += gaussian_loglik(mouse['data'][:, 1], pred, std)
    return ll if np.isfinite(ll) else -np.inf


def log_probability(theta, l_bound, u_bound, model, mice, upstream):
    if any(not (lo < v < hi) for lo, v, hi in zip(l_bound, theta, u_bound)):
        return -np.inf
    return log_likelihood(theta, model, mice, upstream)
