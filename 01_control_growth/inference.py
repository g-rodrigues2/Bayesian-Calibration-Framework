"""Log-prior, log-likelihood and log-probability for the MCMC.

A single ``log_likelihood`` covers both calibration modes through ``mouse_index``:

  mouse_index=None  (global mode)
      theta = [par1..parN, T0_1..T0_M, std], M = mice in the group.
      Growth parameters shared by the group, one free initial condition per mouse and a single
      std; the likelihood is summed over all mice.

  mouse_index=i     (mouse-specific mode)
      theta = [par1..parN, T0, std]; one mouse is fitted.

Both cases use ``_split_theta``; only ``ndim`` (the number of free initial conditions) changes.
"""
import numpy as np

from models import _split_theta, solve_model


def log_likelihood(theta, model_extension, full_data, group, mouse_index=None):
    """Sum of Gaussian log-likelihoods over the target mice.

    If the ODE returns NaN or raises (pathological parameters, e.g. _bert with a dominant b), -inf is
    returned, so that emcee rejects the proposal without stopping the sampler.
    """
    ll = 0.0
    variance = theta[-1] ** 2

    if mouse_index is None:
        mice = full_data[group]
    else:
        mice = [full_data[group][mouse_index]]

    n_ic = len(mice)
    ode_args, ic_fn = _split_theta(theta, model_extension, n_ic)

    for idx, mouse_data in enumerate(mice):
        time_array = mouse_data['data'][:, 0]
        initial_condition = ic_fn(idx)
        try:
            solution = solve_model(model_extension, time_array, ode_args,
                                   initial_condition)
        except Exception:
            return -np.inf

        sol_T = solution[:, 0]
        if not np.all(np.isfinite(sol_T)):
            return -np.inf

        observed_volume = mouse_data['data'][:, 1]
        ll += -0.5 * np.sum(
            (sol_T - observed_volume) ** 2 / variance
            + np.log(2 * np.pi) + np.log(variance)
        )

    if not np.isfinite(ll):
        return -np.inf
    return ll


def log_likelihood_pointwise(theta, model_extension, full_data, group,
                             mouse_index=None):
    """log p(y_i | theta) for each observation i, concatenated over mice.

    Returns an array of length n_obs, or None if the ODE fails. Its sum equals `log_likelihood(...)`;
    WAIC and PSIS-LOO need the pointwise values because they assess predictive accuracy observation by
    observation instead of summarizing the posterior by a single theta (which is where DIC fails).
    """
    variance = theta[-1] ** 2
    if not np.isfinite(variance) or variance <= 0:
        return None

    mice = full_data[group] if mouse_index is None else [full_data[group][mouse_index]]
    ode_args, ic_fn = _split_theta(theta, model_extension, len(mice))

    pieces = []
    for idx, mouse_data in enumerate(mice):
        time_array = mouse_data['data'][:, 0]
        try:
            solution = solve_model(model_extension, time_array, ode_args,
                                   ic_fn(idx))
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


def log_prior(theta, l_bound, u_bound):
    """Uniform prior on the box [l_bound, u_bound]: 0 inside, -inf outside."""
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0


def log_probability(theta, l_bound, u_bound, model_extension, full_data, group,
                    mouse_index=None):
    """Unnormalized log-posterior = log_prior + log_likelihood."""
    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model_extension, full_data, group,
                               mouse_index)
