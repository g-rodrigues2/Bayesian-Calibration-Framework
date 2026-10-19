"""Multi-group likelihood of the radiotherapy sharing scenarios.

Sum over both groups, with growth (r, cc) fixed per group, the radiation effect anchored at the dose
days, and the radiotherapy parameters (alpha, beta, delay) shared or group-specific as set by the
scenario. std is global.
"""
import _path_setup  # noqa: F401

import os

import numpy as np

from radiation_models import solve_model_dispatch
from scenarios_rad import (
    MODEL_PARAM_NAMES, count_model_params, ic_slice_for_group,
    split_model_args_for_group,
)

# Dose per fraction (Gy), from CALIB_DOSE; read at import time (inherited by spawned workers).
DOSE = float(os.environ.get('CALIB_DOSE', '2'))


def log_likelihood_shared(theta, model_extension, full_data, groups, scenario,
                          growth, dose=DOSE):
    std = theta[-1]
    variance = std ** 2
    if not np.isfinite(variance) or variance <= 0:
        return -np.inf

    param_names = MODEL_PARAM_NAMES[model_extension]
    n_model = count_model_params(scenario, groups)

    ll = 0.0
    for group in groups:
        rad_args, _ = split_model_args_for_group(theta, scenario, param_names,
                                                 groups, group)
        T0s = ic_slice_for_group(theta, n_model, full_data, groups, group)
        r, cc = growth[group]['r'], growth[group]['cc']
        for idx, mouse_data in enumerate(full_data[group]):
            try:
                sol = solve_model_dispatch(
                    model_extension, mouse_data['data'][:, 0], (r, cc, *rad_args),
                    float(T0s[idx]), mouse_data['treatment_days'], dose)
            except Exception:
                return -np.inf
            sol_T = sol[:, 0]
            if not np.all(np.isfinite(sol_T)):
                return -np.inf
            obs = mouse_data['data'][:, 1]
            ll += -0.5 * np.sum((sol_T - obs) ** 2 / variance
                                + np.log(2 * np.pi) + np.log(variance))
    return ll if np.isfinite(ll) else -np.inf


def log_likelihood_shared_pointwise(theta, model_extension, full_data, groups,
                                    scenario, growth, dose=DOSE):
    std = theta[-1]
    variance = std ** 2
    if not np.isfinite(variance) or variance <= 0:
        return None
    param_names = MODEL_PARAM_NAMES[model_extension]
    n_model = count_model_params(scenario, groups)
    pieces = []
    for group in groups:
        rad_args, _ = split_model_args_for_group(theta, scenario, param_names,
                                                 groups, group)
        T0s = ic_slice_for_group(theta, n_model, full_data, groups, group)
        r, cc = growth[group]['r'], growth[group]['cc']
        for idx, mouse_data in enumerate(full_data[group]):
            try:
                sol = solve_model_dispatch(
                    model_extension, mouse_data['data'][:, 0], (r, cc, *rad_args),
                    float(T0s[idx]), mouse_data['treatment_days'], dose)
            except Exception:
                return None
            sol_T = sol[:, 0]
            if not np.all(np.isfinite(sol_T)):
                return None
            obs = mouse_data['data'][:, 1]
            pieces.append(-0.5 * ((sol_T - obs) ** 2 / variance
                                   + np.log(2 * np.pi) + np.log(variance)))
    out = np.concatenate(pieces)
    return out if np.all(np.isfinite(out)) else None


def log_prior_uniform(theta, l_bound, u_bound):
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0


def log_probability_shared(theta, l_bound, u_bound, model_extension, full_data,
                           groups, scenario, growth, dose=DOSE):
    lp = log_prior_uniform(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_shared(theta, model_extension, full_data, groups,
                                      scenario, growth, dose)
