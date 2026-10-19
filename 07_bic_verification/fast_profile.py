"""Exact profile likelihood of the radiotherapy BIC cells, for fast maximization.

Every continuous radiotherapy model has the form dT/dt = T [r (1 - T/K) - g(t)], where the effect g(t)
depends only on time and on the radiotherapy parameters. This is a Bernoulli equation: with
G(t) = r (t - t0) - int_t0^t g and S(t) = int_t0^t exp(G),

    T(t) = T0 A(t) / (1 + T0 B(t)),      A = exp(G),   B = (r / K) S,

so A and B do not depend on the initial volume T0. For fixed radiotherapy parameters, the Gaussian
log-likelihood of a group is maximized
  - over each T0 separately (each T0 enters only the residuals of its own mouse), without solving any
    ODE, and
  - over the common std in closed form, std^2 = SSE / n (clipped to the prior box).
The maximum log-likelihood of a cell is therefore the maximum over the radiotherapy parameters alone
(2 to 6 of them instead of up to 30) of this profile, which is what `profile_problem` returns.

A and B are integrated with the classical Runge-Kutta scheme on steps of at most `STEP` days, restarted at
the onset of every fraction effect (tau_i or tau_i + delay), where g jumps, and ending exactly at the
observation times. The effect g(t) is the compiled right-hand side of radiation_models evaluated with
r = 0, K = 1 and T = 1, so the models are not written twice. The linear-quadratic model (discrete
surviving fractions) is not covered.
"""
import numpy as np
from numba import njit

import _stage_paths  # noqa: F401
import radiation_models as RM

STEP = 0.05                     # days; the log-likelihood changes by less than 1e-6 against STEP = 0.01
_GRID = 400                     # grid points of the initial-volume search, refined by golden section
_GOLDEN_ITERS = 60
_KERNELS = {}


def _kernel(model):
    """Compiled integrator of (A, S) at the knots for one model."""
    if model in _KERNELS:
        return _KERNELS[model]
    fn = getattr(RM, 'model' + model)
    n = RM.n_radiation_params(model)
    if n == 2:
        @njit
        def g(t, p, tau, dose):
            return -fn(1.0, t, 0.0, 1.0, p[0], p[1], tau, dose)
    elif n == 3:
        @njit
        def g(t, p, tau, dose):
            return -fn(1.0, t, 0.0, 1.0, p[0], p[1], p[2], tau, dose)
    elif n == 4:
        @njit
        def g(t, p, tau, dose):
            return -fn(1.0, t, 0.0, 1.0, p[0], p[1], p[2], p[3], tau, dose)
    else:
        raise ValueError(f'{model}: {n} radiotherapy parameters')

    @njit
    def advance(G, s, a, b, r, p, tau, dose, h):
        m = max(1, int(np.ceil((b - a) / h - 1e-9)))
        dt = (b - a) / m
        for i in range(m):
            t = a + i * dt
            t_end = b - 1e-9 if i == m - 1 else t + dt   # the jump at b belongs to the next segment
            f1 = r - g(t, p, tau, dose)
            f2 = r - g(t + 0.5 * dt, p, tau, dose)
            f4 = r - g(t_end, p, tau, dose)
            e1 = np.exp(G)
            e2 = np.exp(G + 0.5 * dt * f1)
            e3 = np.exp(G + 0.5 * dt * f2)
            e4 = np.exp(G + dt * f2)
            s += dt / 6.0 * (e1 + 2.0 * e2 + 2.0 * e3 + e4)
            G += dt / 6.0 * (f1 + 4.0 * f2 + f4)
        return G, s

    @njit
    def integrate(r, p, tau, dose, times, onsets, h):
        """A and S at the sorted distinct times; the integration restarts at every onset."""
        nk = times.size
        A = np.empty(nk)
        S = np.empty(nk)
        G = 0.0
        s = 0.0
        A[0] = 1.0
        S[0] = 0.0
        t = times[0]
        j = 0
        while j < onsets.size and onsets[j] <= t:
            j += 1
        for k in range(1, nk):
            target = times[k]
            while j < onsets.size and onsets[j] < target:
                G, s = advance(G, s, t, onsets[j], r, p, tau, dose, h)
                t = onsets[j]
                j += 1
            G, s = advance(G, s, t, target, r, p, tau, dose, h)
            t = target
            while j < onsets.size and onsets[j] <= t:
                j += 1
            A[k] = np.exp(G)
            S[k] = s
        return A, S

    _KERNELS[model] = integrate
    return integrate


@njit
def _sse(T0, A, B, y):
    total = 0.0
    for j in range(y.size):
        d = T0 * A[j] / (1.0 + T0 * B[j]) - y[j]
        total += d * d
    return total


@njit
def best_initial_volume(A, B, y, lo, hi):
    """(T0, SSE) minimizing the squared residuals of one mouse over lo < T0 < hi."""
    best_x, best_f = lo, np.inf
    step = (hi - lo) / _GRID
    for i in range(_GRID + 1):
        x = lo + i * step
        f = _sse(x, A, B, y)
        if f < best_f:
            best_x, best_f = x, f
    a = max(lo, best_x - step)
    b = min(hi, best_x + step)
    ratio = 0.6180339887498949
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc = _sse(c, A, B, y)
    fd = _sse(d, A, B, y)
    for _ in range(_GOLDEN_ITERS):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = _sse(c, A, B, y)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = _sse(d, A, B, y)
    for x, f in ((c, fc), (d, fd)):
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f


@njit
def group_sse(A, S, r_over_K, index, obs, n_obs, lo, hi, T0s):
    """Summed squared residuals of a group at the best initial volumes (written to T0s)."""
    total = 0.0
    for m in range(n_obs.size):
        n = n_obs[m]
        Am = np.empty(n)
        Bm = np.empty(n)
        for j in range(n):
            Am[j] = A[index[m, j]]
            Bm[j] = r_over_K * S[index[m, j]]
        x, f = best_initial_volume(Am, Bm, obs[m, :n], lo[m], hi[m])
        T0s[m] = x
        total += f
    return total


def _onsets(model, p, tau):
    """Times at which g jumps: fraction days, shifted by the delay for the delayed models."""
    if model.endswith('Delay'):
        delay = p[RM._radiation_param_names(model).index('delay')]
        return tau + delay
    return tau


def covered(cell):
    return cell['kind'] in ('radiation', 'radiation_sharing') and cell['model'] not in RM.DISCRETE_KILL_MODELS


def profile_problem(cell, lb, ub):
    """(profile(theta_r) -> loglik, full(theta_r) -> theta, n_r) for a covered cell.

    theta_r is the leading block of theta (radiotherapy parameters); `full` returns the complete vector
    (radiotherapy parameters, maximizing initial volumes, maximizing std)."""
    from likelihoods import _mouse_index, _radiation
    data, _, growth = _radiation()
    model, kind = cell['model'], cell['kind']
    integrate = _kernel(model)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    eps = 1e-9 * (ub - lb)

    if kind == 'radiation':
        block = cell['block']
        mice = data[block] if cell['row'] == 'Global' else [data[block][_mouse_index(data[block], cell['row'])]]
        n_r = RM.n_radiation_params(model)
        blocks = [(block, mice, lambda th: np.asarray(th, float))]
    else:
        from scenarios_rad import MODEL_PARAM_NAMES, SCENARIOS, count_model_params, split_model_args_for_group
        groups = ['radiation_sensitive', 'radiation_resistant']
        scenario = SCENARIOS[model][cell['scenario']]
        names = MODEL_PARAM_NAMES[model]
        n_r = count_model_params(scenario, groups)
        blocks = [(g, data[g], (lambda th, g=g: np.asarray(
            split_model_args_for_group(th, scenario, names, groups, g)[0], float))) for g in groups]

    prepared, offset = [], n_r
    for group, mice, args_of in blocks:
        tau = np.asarray(mice[0]['treatment_days'], float)
        if any(not np.array_equal(np.asarray(m['treatment_days'], float), tau) for m in mice):
            raise ValueError('mice of a group with different fraction days')
        times = [np.asarray(m['data'][:, 0], float) for m in mice]
        if any(t[0] != times[0][0] for t in times):
            raise ValueError('mice of a group with different initial times')
        grid = np.unique(np.concatenate(times))
        n_max = max(t.size for t in times)
        index = np.zeros((len(mice), n_max), dtype=np.int64)
        obs = np.zeros((len(mice), n_max))
        for i, (t, m) in enumerate(zip(times, mice)):
            index[i, :t.size] = np.searchsorted(grid, t)
            obs[i, :t.size] = m['data'][:, 1]
        n_obs_m = np.array([t.size for t in times], dtype=np.int64)
        lo = np.array([lb[offset + i] + eps[offset + i] for i in range(len(mice))])
        hi = np.array([ub[offset + i] - eps[offset + i] for i in range(len(mice))])
        prepared.append((growth[group]['r'], growth[group]['cc'], tau, np.full(tau.size, RM.DOSE_DEFAULT),
                         grid, index, obs, n_obs_m, lo, hi, args_of))
        offset += len(mice)
    n_obs = int(sum(int(p[7].sum()) for p in prepared))
    std_lo, std_hi = lb[-1] + eps[-1], ub[-1] - eps[-1]

    def solve(theta_r):
        sse, T0s = 0.0, []
        for r, K, tau, dose, grid, index, obs, n_obs_m, lo, hi, args_of in prepared:
            p = args_of(theta_r)
            onsets = np.sort(_onsets(model, p, tau))
            A, S = integrate(r, p, tau, dose, grid, onsets, STEP)
            x = np.empty(n_obs_m.size)
            sse += group_sse(A, S, r / K, index, obs, n_obs_m, lo, hi, x)
            T0s.extend(x.tolist())
        std = float(np.clip(np.sqrt(sse / n_obs), std_lo, std_hi))
        ll = -0.5 * (sse / std ** 2 + n_obs * np.log(2.0 * np.pi * std ** 2))
        return ll, T0s, std

    def profile(theta_r):
        ll = solve(theta_r)[0]
        return ll if np.isfinite(ll) else -np.inf

    def full(theta_r):
        _, T0s, std = solve(theta_r)
        return np.concatenate([np.asarray(theta_r, float), T0s, [std]])

    return profile, full, n_r
