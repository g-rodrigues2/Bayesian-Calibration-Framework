"""Detection of stuck walkers from the per-walker log-probability.

Motivation (measured in the TNBC data, _log model, control_resistant, 10000 steps): about 7% of the
post-burn-in samples came from walkers trapped in a spurious high-`std` mode (std ~ 490 against 111;
log L ~ -810 against -642). A criterion based on

    ll_spread = ll_max - ll_med > 50

gave 4.05 in that case and did not flag anything, because the tail lies below the median and the
maximum is in the good mode. A lower-tail criterion

    ll_tail = ll_med - percentile_5(ll)

gave 168 and flags it. Both columns are reported; the flag uses `ll_tail`.

emcee keeps the walker identity in the HDF5 backend (log_prob has shape (n_iter, nwalkers)), so the
stuck walkers themselves can be identified by their mean log-likelihood.
"""
import warnings

import numpy as np

# A walker whose mean log L lies more than TOL nats below the median walker is in another regime,
# not merely exploring the tail.
STUCK_TOL = 50.0
# If too many walkers are flagged, the whole chain has not converged: nothing is removed, so that the
# diagnostic is not masked.
MAX_STUCK_FRAC = 0.5


def ll_tail(log_prob_flat, q=5.0):
    """Median minus the q-th percentile: size of the lower tail of log L."""
    lp = np.asarray(log_prob_flat, float)
    lp = lp[np.isfinite(lp)]
    if lp.size == 0:
        return float('nan')
    return float(np.median(lp) - np.percentile(lp, q))


def walker_mean_loglik(log_prob):
    """Mean log L of each walker, ignoring non-finite samples.

    log_prob has shape (n_iter, nwalkers). A walker without any finite sample returns NaN (and is
    treated as stuck by `find_stuck_walkers`).
    """
    lp = np.asarray(log_prob, float)
    if lp.ndim != 2:
        raise ValueError(f'expected (n_iter, nwalkers), got shape {lp.shape}')
    masked = np.where(np.isfinite(lp), lp, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN walker
        return np.nanmean(masked, axis=0)


def find_stuck_walkers(log_prob, tol=STUCK_TOL, max_frac=MAX_STUCK_FRAC):
    """Boolean mask (nwalkers,) of the stuck walkers.

    Stuck = mean log L below `median over walkers - tol`. If the stuck fraction exceeds `max_frac`,
    all False is returned: there is no reference good mode and removal would be arbitrary.
    """
    means = walker_mean_loglik(log_prob)
    finite = np.isfinite(means)
    if not finite.any():
        return np.zeros(len(means), dtype=bool)

    ref = np.median(means[finite])
    stuck = np.zeros(len(means), dtype=bool)
    stuck[finite] = means[finite] < ref - tol
    stuck[~finite] = True          # a walker without finite log L is also removed

    if stuck.sum() > max_frac * len(means):
        return np.zeros(len(means), dtype=bool)
    return stuck


def drop_stuck(chain, log_prob, tol=STUCK_TOL, max_frac=MAX_STUCK_FRAC):
    """Removes the stuck walkers from non-flattened (chain, log_prob).

    chain:    (n_iter, nwalkers, ndim)
    log_prob: (n_iter, nwalkers)
    Returns (filtered chain, filtered log_prob, stuck mask).
    """
    stuck = find_stuck_walkers(log_prob, tol=tol, max_frac=max_frac)
    keep = ~stuck
    return chain[:, keep, :], log_prob[:, keep], stuck
