"""Log-prior, log-likelihood and log-probability of the immunotherapy calibration.

Growth (r, cc) is fixed per group (inherited from the control groups); only the treatment parameters
are calibrated. The effect is anchored at the administration days.

A single log_likelihood covers both modes through `mouse_index`:
  None -> whole group. theta = [<parameters (global and/or per mouse)>, ic_1..ic_N, std]
  i    -> one mouse.   theta = [<parameters>, ic, std]

The group mode supports a mixed parameterization: `ms_params` lists the parameters calibrated per mouse
(the others are global). See bounds_imm.
"""
import numpy as np

from bounds_imm import extract_treatment_params, treatment_param_names
from immuno_models import DOSE_DEFAULT, solve_model


def _params_and_ic(theta, model_extension, mice, mouse_index, ms_params):
    """(fn_params(idx), T0s) for the mode and mixed parameterization."""
    if mouse_index is None:
        n = len(mice)
        n_slots = None

        def fn(idx):
            p, slots = extract_treatment_params(theta, model_extension, n, idx,
                                                ms_params)
            return p, slots
        _, n_slots = fn(0)
        return fn, theta[n_slots:n_slots + n]

    # one mouse: every parameter is a single entry
    n_p = len(treatment_param_names(model_extension))

    def fn1(_idx):
        return tuple(theta[:n_p]), n_p
    return fn1, theta[n_p:n_p + 1]


def log_likelihood(theta, model_extension, full_data, group, growth,
                   mouse_index=None, ms_params=(), dose=DOSE_DEFAULT):
    variance = theta[-1] ** 2
    if not np.isfinite(variance) or variance <= 0:
        return -np.inf

    mice = (full_data[group] if mouse_index is None
            else [full_data[group][mouse_index]])
    fn, T0s = _params_and_ic(theta, model_extension, full_data[group],
                             mouse_index, ms_params)
    r, cc = growth[group]['r'], growth[group]['cc']

    ll = 0.0
    for i, mouse in enumerate(mice):
        idx = mouse_index if mouse_index is not None else i
        params, _ = fn(idx if mouse_index is None else 0)
        T0 = float(T0s[i if mouse_index is None else 0])
        try:
            sol = solve_model(model_extension, mouse['data'][:, 0],
                              (r, cc, *params), T0, mouse['treatment_days'], dose)
        except Exception:
            return -np.inf
        sol_T = sol[:, 0]
        if not np.all(np.isfinite(sol_T)):
            return -np.inf
        obs = mouse['data'][:, 1]
        ll += -0.5 * np.sum((sol_T - obs) ** 2 / variance
                            + np.log(2 * np.pi) + np.log(variance))
    return ll if np.isfinite(ll) else -np.inf


def log_likelihood_pointwise(theta, model_extension, full_data, group, growth,
                             mouse_index=None, ms_params=(), dose=DOSE_DEFAULT):
    """log p(y_i|theta) of each observation (input of WAIC/LOO)."""
    variance = theta[-1] ** 2
    if not np.isfinite(variance) or variance <= 0:
        return None
    mice = (full_data[group] if mouse_index is None
            else [full_data[group][mouse_index]])
    fn, T0s = _params_and_ic(theta, model_extension, full_data[group],
                             mouse_index, ms_params)
    r, cc = growth[group]['r'], growth[group]['cc']

    pieces = []
    for i, mouse in enumerate(mice):
        idx = mouse_index if mouse_index is not None else i
        params, _ = fn(idx if mouse_index is None else 0)
        T0 = float(T0s[i if mouse_index is None else 0])
        try:
            sol = solve_model(model_extension, mouse['data'][:, 0],
                              (r, cc, *params), T0, mouse['treatment_days'], dose)
        except Exception:
            return None
        sol_T = sol[:, 0]
        if not np.all(np.isfinite(sol_T)):
            return None
        obs = mouse['data'][:, 1]
        pieces.append(-0.5 * ((sol_T - obs) ** 2 / variance
                               + np.log(2 * np.pi) + np.log(variance)))
    out = np.concatenate(pieces)
    return out if np.all(np.isfinite(out)) else None


def log_prior(theta, l_bound, u_bound):
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0


def to_sampling(theta, log_index):
    """Parameter vector(s) -> sampling coordinates (log of the entries in log_index)."""
    phi = np.array(theta, dtype=float, copy=True)
    phi[..., log_index] = np.log(phi[..., log_index])
    return phi


def from_sampling(phi, log_index):
    """Sampling coordinates -> parameter vector(s) (inverse of to_sampling)."""
    theta = np.array(phi, dtype=float, copy=True)
    theta[..., log_index] = np.exp(theta[..., log_index])
    return theta


def log_probability_logscale(phi, l_bound, u_bound, log_index, model_extension, full_data, group,
                             growth, mouse_index=None, ms_params=(), dose=DOSE_DEFAULT):
    """Log-posterior in sampling coordinates where the entries in log_index are log-transformed.

    The Jacobian term sum(phi[log_index]) keeps the uniform prior of the original parameters, so the
    posterior of theta is the same as with log_probability; only the geometry seen by the sampler
    changes. `l_bound` and `u_bound` are the bounds of the original parameters.
    """
    theta = from_sampling(phi, log_index)
    lp = log_probability(theta, l_bound, u_bound, model_extension, full_data, group, growth,
                         mouse_index, ms_params, dose)
    return lp + float(np.sum(np.asarray(phi)[log_index])) if np.isfinite(lp) else -np.inf


def log_probability(theta, l_bound, u_bound, model_extension, full_data, group,
                    growth, mouse_index=None, ms_params=(), dose=DOSE_DEFAULT):
    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model_extension, full_data, group, growth,
                               mouse_index, ms_params, dose)
