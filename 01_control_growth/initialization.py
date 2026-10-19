"""Initial positions of the emcee walkers.

The original code spread the walkers uniformly over the prior box:

    pos[:, i] = np.random.uniform(low=l_bound[i], high=u_bound[i], size=nwalkers)

This is inefficient when the box is much larger than the region of posterior mass. Example measured
in the TNBC data (_log model, control_resistant): the bound of `std` is [0.1, 1000], so half of the
walkers start with std > 500, where the likelihood is almost flat. With the stretch move this subset
may never be recovered: about 7% of the post-burn-in samples remained at std ~ 490 and log L ~ -810
(against -642 in the good mode), inflating Var(log L) by a factor of 400.

AIC, AICc and BIC use only the maximum log-likelihood and are not affected, but DIC (the posterior
mean leaves the ridge, p_D < 0) and WBIC (importance weights dominated by the worst sample, ESS ~ 3)
are.

Standard emcee practice is to initialize in a small ball around a good point.
`initial_positions(mode='ball')` finds an approximate MAP (random sampling followed by Powell inside
the box) and spreads the walkers around it with relative deviation `scale`. `mode='uniform'`
reproduces the original behavior.
"""
import numpy as np
from scipy.optimize import minimize


def uniform_positions(nwalkers, l_bound, u_bound, rng):
    """Uniform spread over the prior box (original behavior)."""
    l, u = np.asarray(l_bound, float), np.asarray(u_bound, float)
    return rng.uniform(l, u, size=(nwalkers, len(l)))


def find_map_estimate(log_prob, args, l_bound, u_bound, n_random=200, rng=None,
                      maxiter=2000, maxfev=None):
    """Approximate MAP: best of n_random draws, refined by Powell.

    Powell is gradient-free and accepts `bounds`, so the search never leaves the box (where log_prior
    is -inf). Returns None if no draw has a finite log-probability.

    `maxfev` limits the number of function evaluations of the refinement, which matters when log_prob
    is expensive (ODEs with dose schedules). None = no limit.
    """
    rng = rng or np.random.default_rng()
    l, u = np.asarray(l_bound, float), np.asarray(u_bound, float)

    cand = rng.uniform(l, u, size=(n_random, len(l)))
    lps = np.array([log_prob(t, *args) for t in cand])
    if not np.isfinite(lps).any():
        return None
    theta0 = cand[int(np.nanargmax(lps))]

    eps = 1e-9 * (u - l)
    opts = {'maxiter': maxiter, 'xtol': 1e-6, 'ftol': 1e-6}
    if maxfev is not None:
        opts['maxfev'] = int(maxfev)
    try:
        res = minimize(lambda t: -log_prob(t, *args), theta0, method='Powell',
                       bounds=list(zip(l + eps, u - eps)), options=opts)
        theta = np.clip(np.asarray(res.x, float), l + eps, u - eps)
        if np.isfinite(log_prob(theta, *args)) and -res.fun >= lps.max():
            return theta
    except Exception:
        pass
    return theta0


def ball_positions(theta0, nwalkers, l_bound, u_bound, log_prob, args,
                   scale=1e-3, rng=None, max_tries=100):
    """Walkers in a relative Gaussian ball around theta0.

    The deviation of each coordinate is `scale * max(|theta0_i|, width_i * 1e-6)`, so the ball respects
    the scale of each parameter (r ~ 0.1 and cc ~ 4000 get proportional spreads). Any walker with a
    non-finite log_prob is redrawn.
    """
    rng = rng or np.random.default_rng()
    l, u = np.asarray(l_bound, float), np.asarray(u_bound, float)
    theta0 = np.asarray(theta0, float)
    eps = 1e-9 * (u - l)

    width = np.maximum(np.abs(theta0), (u - l) * 1e-6) * scale
    pos = np.clip(theta0 + width * rng.normal(size=(nwalkers, len(theta0))),
                  l + eps, u - eps)

    for i in range(nwalkers):
        tries = 0
        while not np.isfinite(log_prob(pos[i], *args)) and tries < max_tries:
            pos[i] = np.clip(theta0 + width * rng.normal(size=len(theta0)),
                             l + eps, u - eps)
            tries += 1
    return pos


def initial_positions(nwalkers, l_bound, u_bound, log_prob, args,
                      mode='ball', scale=1e-3, n_random=200, seed=None,
                      map_maxiter=2000, map_maxfev=None):
    """Returns (pos, info). mode: 'ball' (default) or 'uniform'.

    If no MAP is found (no finite draw), falls back to 'uniform'. `info` holds the mode actually used
    and the log_prob of the center. `map_maxfev` limits the Powell evaluations; use a small value
    (e.g. 300) when log_prob is expensive.
    """
    rng = np.random.default_rng(seed)
    if mode == 'uniform':
        return uniform_positions(nwalkers, l_bound, u_bound, rng), \
            {'mode': 'uniform', 'center_log_prob': None}

    if mode != 'ball':
        raise ValueError(f"invalid CALIB_INIT: {mode!r} (use 'ball' or 'uniform')")

    theta0 = find_map_estimate(log_prob, args, l_bound, u_bound,
                               n_random=n_random, rng=rng,
                               maxiter=map_maxiter, maxfev=map_maxfev)
    if theta0 is None:
        return uniform_positions(nwalkers, l_bound, u_bound, rng), \
            {'mode': 'uniform (fallback: no MAP estimate found)',
             'center_log_prob': None}

    pos = ball_positions(theta0, nwalkers, l_bound, u_bound, log_prob, args,
                         scale=scale, rng=rng)
    return pos, {'mode': 'ball', 'center_log_prob': float(log_prob(theta0, *args))}
