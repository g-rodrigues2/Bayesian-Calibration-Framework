"""Log-prior, log-likelihood and log-probability of the radiotherapy calibration.

Growth (r, cc) is fixed per group (growth_params); only the radiotherapy parameters are calibrated.
The treatment effect is anchored at the dose days of each mouse (`treatment_days`).

A single log_likelihood covers both modes through `mouse_index`:
  None -> whole group (theta = [rad..., ic1..icM, std])
  i    -> one mouse   (theta = [rad..., ic, std])
"""
import os

import numpy as np

from radiation_models import solve_model_dispatch, n_radiation_params

# Dose per fraction (Gy), from CALIB_DOSE; read at import time (inherited by spawned workers).
DOSE = float(os.environ.get('CALIB_DOSE', '2'))


def log_likelihood(theta, model_extension, full_data, group, growth,
                   mouse_index=None, dose=DOSE):
    """Sum of Gaussian log-likelihoods; `growth` = {group: {r, cc}}.

    NaN or an exception in the ODE solution returns -inf (emcee rejects the proposal).
    """
    variance = theta[-1] ** 2
    if not np.isfinite(variance) or variance <= 0:
        return -np.inf

    mice = full_data[group] if mouse_index is None else [full_data[group][mouse_index]]
    n_ic = len(mice)
    n_rad = n_radiation_params(model_extension)
    rad_params = tuple(theta[:n_rad])
    T0s = theta[n_rad:n_rad + n_ic]
    r = growth[group]['r']
    cc = growth[group]['cc']

    ll = 0.0
    for idx, mouse_data in enumerate(mice):
        time_array = mouse_data['data'][:, 0]
        try:
            solution = solve_model_dispatch(
                model_extension, time_array, (r, cc, *rad_params),
                float(T0s[idx]), mouse_data['treatment_days'], dose)
        except Exception:
            return -np.inf
        sol_T = solution[:, 0]
        if not np.all(np.isfinite(sol_T)):
            return -np.inf
        observed = mouse_data['data'][:, 1]
        ll += -0.5 * np.sum((sol_T - observed) ** 2 / variance
                            + np.log(2 * np.pi) + np.log(variance))

    return ll if np.isfinite(ll) else -np.inf


def log_prior(theta, l_bound, u_bound):
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0


def log_probability(theta, l_bound, u_bound, model_extension, full_data, group,
                    growth, mouse_index=None, dose=DOSE):
    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model_extension, full_data, group, growth,
                               mouse_index, dose)


def log_likelihood_pointwise(theta, model_extension, full_data, group, growth,
                             mouse_index=None, dose=DOSE):
    """log p(y_i|theta) of each observation (input of WAIC/LOO); None if the ODE solution fails."""
    variance = theta[-1] ** 2
    if not np.isfinite(variance) or variance <= 0:
        return None
    mice = full_data[group] if mouse_index is None else [full_data[group][mouse_index]]
    n_rad = n_radiation_params(model_extension)
    rad_params = tuple(theta[:n_rad])
    T0s = theta[n_rad:n_rad + len(mice)]
    r, cc = growth[group]['r'], growth[group]['cc']

    pieces = []
    for idx, mouse_data in enumerate(mice):
        try:
            solution = solve_model_dispatch(
                model_extension, mouse_data['data'][:, 0], (r, cc, *rad_params),
                float(T0s[idx]), mouse_data['treatment_days'], dose)
        except Exception:
            return None
        sol_T = solution[:, 0]
        if not np.all(np.isfinite(sol_T)):
            return None
        obs = mouse_data['data'][:, 1]
        pieces.append(-0.5 * ((sol_T - obs) ** 2 / variance
                              + np.log(2 * np.pi) + np.log(variance)))
    out = np.concatenate(pieces)
    return out if np.all(np.isfinite(out)) else None
