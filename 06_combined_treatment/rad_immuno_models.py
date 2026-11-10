"""Full radiotherapy + immunotherapy model (Eqs. 4.2 and 4.10 of the thesis).

    dT/dt = r T (1 - T/K)
            - a_rad T sum_i D_i exp(-b_rad (t - tau_i - delay)) H(t - tau_i - delay)
            - T sum_j phi D_j^lam / (s T^lam + D_j^lam) H(t - tau_j)

All models share this equation and differ only in the lysis coefficient phi, which is constant in
time for each mouse:

    _RadImmunoFixed       phi = d                    (without SUV, d of the immunotherapy cohort)
    _RadImmunoFitD        phi = d                    (without SUV, d estimated with the data at hand)
    _SUVLinear            phi = a SUV + b
    _SUVMichaelisMenten   phi = a SUV / (b + SUV)
    _SUVLogistic          phi = a / (1 + exp(-b SUV))
    _SUVSatExp            phi = a (1 - exp(-b SUV))
    _SUVExp               phi = exp(a z + b),  z = standardized SUV
    _VolExp               phi = exp(a z + b),  z = standardized initial tumor volume

The last two keep phi positive for every value of the coefficients, which the linear form does not,
and standardize the covariate within each training fold, so that the two coefficients are on the same
scale. _VolExp replaces the imaging covariate by the initial volume and measures how much of the
predictive value of _SUVExp is specific to the imaging measurement.
"""
import numpy as np
from numba import jit
from scipy.integrate import odeint

MODEL_PARAMS = {
    '_RadImmunoFixed': (),
    '_RadImmunoFitD': ('d',),
    '_SUVLinear': ('a', 'b'),
    '_SUVMichaelisMenten': ('a', 'b'),
    '_SUVLogistic': ('a', 'b'),
    '_SUVSatExp': ('a', 'b'),
    '_SUVExp': ('a', 'b'),
    '_VolExp': ('a', 'b'),
}
SUV_MODELS = ('_SUVLinear', '_SUVMichaelisMenten', '_SUVLogistic', '_SUVSatExp')
COVARIATE = {'_SUVExp': 'SUV', '_VolExp': 'volume0'}       # models with a standardized covariate


def covariate_value(model, mouse):
    """Raw covariate of one mouse: the first observed volume for _VolExp, the PET uptake otherwise."""
    return float(mouse['data'][0, 1]) if COVARIATE.get(model) == 'volume0' else mouse['SUV']


def standardized_upstream(model, upstream, training):
    """upstream with the mean and standard deviation of the covariate over the training mice.

    The standardization never sees the held-out mouse, so every quantity needed for a strict
    out-of-sample prediction comes from the training fold alone."""
    if model not in COVARIATE:
        return upstream
    values = np.array([covariate_value(model, m) for m in training])
    spread = float(values.std(ddof=1))
    if not spread > 0:
        raise ValueError(f'Covariate {COVARIATE[model]!r} is constant across the training mice.')
    return {**upstream, 'cov_mean': float(values.mean()), 'cov_std': spread}


@jit(nopython=True)
def rhs(y, t, r, cc, a_rad, b_rad, delay, phi, s, lam, tau_rad, tau_io, dose_rad, dose_io):
    tumor = max(y[0], 1e-12)
    rad = 0.0
    for i in range(tau_rad.shape[0]):
        if t >= tau_rad[i] + delay:
            rad += a_rad * dose_rad * np.exp(-b_rad * (t - tau_rad[i] - delay))
    io = 0.0
    tumor_pow = tumor ** lam
    for j in range(tau_io.shape[0]):
        if t >= tau_io[j]:
            dose_pow = dose_io ** lam
            io += phi * dose_pow / (s * tumor_pow + dose_pow)
    return r * tumor * (1.0 - tumor / cc) - (rad + io) * tumor


def lysis_coefficient(model, params, covariate, upstream):
    """Lysis coefficient of one mouse; covariate is its SUV, or its initial volume for _VolExp."""
    if model == '_RadImmunoFixed':
        return upstream['d']
    if model == '_RadImmunoFitD':
        return params[0]
    a, b = params
    if model in COVARIATE:
        return np.exp(a * (covariate - upstream['cov_mean']) / upstream['cov_std'] + b)
    suv = covariate
    if model == '_SUVLinear':
        return a * suv + b
    if model == '_SUVMichaelisMenten':
        return a * suv / (b + suv)
    if model == '_SUVLogistic':
        return a / (1.0 + np.exp(-b * suv))
    if model == '_SUVSatExp':
        return a * (1.0 - np.exp(-b * suv))
    raise ValueError(f'Unknown model {model}')


def solve_mouse(model, params, ic, mouse, upstream, dose_rad=2.0, dose_io=1.0, smooth=False):
    """Tumor volume at the observed times, or (t, T) on a 0.1-day grid if smooth=True."""
    t_obs = mouse['data'][:, 0]
    t = np.linspace(t_obs[0], t_obs[-1], int(round((t_obs[-1] - t_obs[0]) / 0.1)) + 1) if smooth else t_obs
    phi = lysis_coefficient(model, params, covariate_value(model, mouse), upstream)
    T = odeint(rhs, [float(ic)], t, mxstep=2000,
               args=(upstream['r'], upstream['cc'], upstream['a_rad'], upstream['b_rad'], upstream['delay'],
                     float(phi), upstream['s'], upstream['lam'], mouse['rad_days'], mouse['io_days'],
                     float(dose_rad), float(dose_io)))[:, 0]
    return (t, T) if smooth else T
