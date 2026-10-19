"""Compiled log-likelihood of the radiotherapy + immunotherapy model (used by the MCMC).

The right-hand side of the model (rad_immuno_models.rhs) is discontinuous at every administration:
each radiotherapy effect switches on at tau_i + delay and each immunotherapy effect at tau_j. Here the
model is integrated in nopython mode with the explicit Dormand-Prince 5(4) pair and adaptive steps,
restarting at each switching time and stopping exactly at each observation time. No step crosses a
discontinuity, no interpolation is needed, and the whole cohort likelihood is evaluated in one
compiled call instead of about 20,000 calls of the right-hand side from scipy's odeint.

Accuracy (posterior samples of the leave-one-out folds, reference DOP853 with rtol=1e-13 restarted at
the same switching times): the largest volume error is 1.4e-6 mm^3 with the default RTOL=1e-9, against
3.2e-3 mm^3 for odeint; log-likelihoods differ from those computed with odeint by less than 3e-4.
"""
import numpy as np
from numba import njit

from rad_immuno_models import lysis_coefficient

RTOL = 1e-9
ATOL = 1e-8
MAX_STEPS = 2_000_000

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


@njit(cache=True)
def _rhs(t, y, r, cc, a_rad, b_rad, delay, dose_rad, tau_rad, n_rad, io_rate, s, lam, dose_pow):
    """rad_immuno_models.rhs with the active administrations fixed: the first n_rad radiotherapy
    fractions and an immunotherapy rate io_rate = phi x (number of active administrations)."""
    tumor = max(y, 1e-12)
    rad = 0.0
    for i in range(n_rad):
        rad += a_rad * dose_rad * np.exp(-b_rad * (t - tau_rad[i] - delay))
    io = 0.0
    if io_rate != 0.0:
        io = io_rate * dose_pow / (s * tumor ** lam + dose_pow)
    return r * tumor * (1.0 - tumor / cc) - (rad + io) * tumor


@njit(cache=True)
def _segment(t, y, t_end, h, r, cc, a_rad, b_rad, delay, dose_rad, tau_rad, n_rad, io_rate, s, lam, dose_pow,
             rtol, atol, budget):
    """Integrates from t to t_end (smooth right-hand side). Returns (y, next step, steps used)."""
    steps = 0
    k1 = _rhs(t, y, r, cc, a_rad, b_rad, delay, dose_rad, tau_rad, n_rad, io_rate, s, lam, dose_pow)
    while t < t_end:
        if steps >= budget:
            return np.nan, h, steps
        last = False
        if t + h >= t_end:
            h_try = t_end - t
            last = True
        else:
            h_try = h
        k2 = _rhs(t + C2 * h_try, y + h_try * A21 * k1, r, cc, a_rad, b_rad, delay, dose_rad, tau_rad, n_rad,
                  io_rate, s, lam, dose_pow)
        k3 = _rhs(t + C3 * h_try, y + h_try * (A31 * k1 + A32 * k2), r, cc, a_rad, b_rad, delay, dose_rad,
                  tau_rad, n_rad, io_rate, s, lam, dose_pow)
        k4 = _rhs(t + C4 * h_try, y + h_try * (A41 * k1 + A42 * k2 + A43 * k3), r, cc, a_rad, b_rad, delay,
                  dose_rad, tau_rad, n_rad, io_rate, s, lam, dose_pow)
        k5 = _rhs(t + C5 * h_try, y + h_try * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4), r, cc, a_rad, b_rad,
                  delay, dose_rad, tau_rad, n_rad, io_rate, s, lam, dose_pow)
        k6 = _rhs(t + h_try, y + h_try * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5), r, cc, a_rad,
                  b_rad, delay, dose_rad, tau_rad, n_rad, io_rate, s, lam, dose_pow)
        y_new = y + h_try * (B1 * k1 + B3 * k3 + B4 * k4 + B5 * k5 + B6 * k6)
        t_new = t_end if last else t + h_try
        k7 = _rhs(t_new, y_new, r, cc, a_rad, b_rad, delay, dose_rad, tau_rad, n_rad, io_rate, s, lam, dose_pow)
        err = h_try * (E1 * k1 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7)
        scale = atol + rtol * max(abs(y), abs(y_new))
        ratio = abs(err) / scale
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


@njit(cache=True)
def predict_mouse(ic, phi, times, n_obs, r, cc, a_rad, b_rad, delay, s, lam, tau_rad, tau_io, dose_rad, dose_io,
                  rtol, atol):
    """Tumor volume at times[:n_obs] (times[0] is the initial time, where the volume is ic)."""
    out = np.empty(n_obs)
    t = times[0]
    y = ic
    out[0] = ic
    tau_rad = np.sort(tau_rad)          # active fractions are then the first n_rad
    switches = np.concatenate((tau_rad + delay, tau_io))
    events = np.sort(np.concatenate((switches, times[1:n_obs])))
    dose_pow = dose_io ** lam
    k = 1
    while k < n_obs and times[k] <= t:
        out[k] = y
        k += 1
    h = 0.1
    budget = MAX_STEPS
    for e in range(events.shape[0]):
        t_end = events[e]
        if t_end <= t:
            continue
        if k >= n_obs:
            break
        n_rad = 0
        for i in range(tau_rad.shape[0]):
            if t >= tau_rad[i] + delay:
                n_rad += 1
        n_io = 0
        for j in range(tau_io.shape[0]):
            if t >= tau_io[j]:
                n_io += 1
        y, h, used = _segment(t, y, t_end, h, r, cc, a_rad, b_rad, delay, dose_rad, tau_rad, n_rad, phi * n_io, s,
                              lam, dose_pow, rtol, atol, budget)
        budget -= used
        if not np.isfinite(y):
            out[k:] = np.nan
            return out
        t = t_end
        while k < n_obs and times[k] <= t:
            out[k] = y
            k += 1
    return out


@njit(cache=True)
def cohort_loglik(ics, phis, std, times, volumes, n_obs, r, cc, a_rad, b_rad, delay, s, lam, tau_rad, tau_io,
                  dose_rad, dose_io, rtol, atol):
    """Gaussian log-likelihood summed over mice (rows of times/volumes, n_obs[m] observations each)."""
    total = 0.0
    var = std * std
    log_norm = np.log(2.0 * np.pi * var)
    for m in range(ics.shape[0]):
        n = n_obs[m]
        pred = predict_mouse(ics[m], phis[m], times[m], n, r, cc, a_rad, b_rad, delay, s, lam, tau_rad, tau_io,
                             dose_rad, dose_io, rtol, atol)
        for k in range(n):
            if not np.isfinite(pred[k]):
                return -np.inf
            d = pred[k] - volumes[m, k]
            total += -0.5 * (d * d / var + log_norm)
    return total if np.isfinite(total) else -np.inf


class Cohort:
    """Packed arrays of a list of mice (same treatment schedule for every mouse)."""

    def __init__(self, mice):
        n_max = max(len(m['data']) for m in mice)
        self.names = tuple(m['name'] for m in mice)
        self.times = np.zeros((len(mice), n_max))
        self.volumes = np.zeros((len(mice), n_max))
        self.n_obs = np.array([len(m['data']) for m in mice], dtype=np.int64)
        for i, m in enumerate(mice):
            self.times[i, :len(m['data'])] = m['data'][:, 0]
            self.volumes[i, :len(m['data'])] = m['data'][:, 1]
        self.suv = np.array([np.nan if m['SUV'] is None else m['SUV'] for m in mice])
        self.tau_rad = np.asarray(mice[0]['rad_days'], dtype=float)
        self.tau_io = np.asarray(mice[0]['io_days'], dtype=float)
        for m in mice:
            if not (np.array_equal(m['rad_days'], self.tau_rad) and np.array_equal(m['io_days'], self.tau_io)):
                raise ValueError('Cohort requires the same treatment schedule for every mouse.')


_COHORTS = {}


def cohort(mice):
    key = tuple(m['name'] for m in mice)
    if key not in _COHORTS:
        _COHORTS[key] = Cohort(mice)
    return _COHORTS[key]


def log_likelihood(model, params, ics, std, mice, upstream, dose_rad=2.0, dose_io=1.0, rtol=RTOL, atol=ATOL):
    c = cohort(mice)
    if model == '_RadImmunoFixed':
        phis = np.full(len(mice), float(upstream['d']))
    else:
        phis = np.array([float(lysis_coefficient(model, params, v, upstream)) for v in c.suv])
    return float(cohort_loglik(np.asarray(ics, dtype=float), phis, float(std), c.times, c.volumes, c.n_obs,
                               upstream['r'], upstream['cc'], upstream['a_rad'], upstream['b_rad'],
                               upstream['delay'], upstream['s'], upstream['lam'], c.tau_rad, c.tau_io,
                               float(dose_rad), float(dose_io), rtol, atol))


def predict(model, params, ic, mouse, upstream, dose_rad=2.0, dose_io=1.0, rtol=RTOL, atol=ATOL):
    """Volumes at the observed times of one mouse (same values as rad_immuno_models.solve_mouse)."""
    t = np.ascontiguousarray(mouse['data'][:, 0], dtype=float)
    phi = float(lysis_coefficient(model, params, mouse['SUV'], upstream))
    return predict_mouse(float(ic), phi, t, len(t), upstream['r'], upstream['cc'], upstream['a_rad'],
                         upstream['b_rad'], upstream['delay'], upstream['s'], upstream['lam'],
                         np.asarray(mouse['rad_days'], dtype=float), np.asarray(mouse['io_days'], dtype=float),
                         float(dose_rad), float(dose_io), rtol, atol)
