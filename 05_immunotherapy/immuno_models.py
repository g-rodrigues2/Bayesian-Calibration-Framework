"""Immunotherapy models and ODE solution with an administration schedule.

dV/dt = r V (1 - V/cc) - I(V, t) V

Growth (r, cc) is inherited from the control groups and fixed (growth_params). The immunotherapy effect
is anchored at the administration days (`tau_immuno`, CALIB_DOSE_DAYS; thesis schedule
0,3,6,9,12,15,18). The dose is normalized (default 1.0 per administration).

The 12 candidate formulations of Table S3.1 of the thesis:
  exponential decay (ED), cumulative dose (CD), accelerated ED and CD (AED, ACD), Michaelis-Menten
  (MM), the de Pillis-Radunskaya law (PR), and ED/CD with logistic, exponential or sinusoidal onset
  factors.
"""
import inspect as _inspect
import os

import numpy as np
from numba import jit
from scipy.integrate import odeint

import compiled_ode  # noqa: E402  (01_control_growth, on sys.path through _path_setup)

# 'compiled' (default): compiled Dormand-Prince solver restarted at the administrations; 'odeint': scipy
SOLVER = os.environ.get('CALIB_SOLVER', 'compiled').lower()
DOSE_DEFAULT = float(os.environ.get('CALIB_DOSE', '1'))


@jit(nopython=True)
def model_ExponentialDecay(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            treatment_effect += a_imuno * dose * np.exp(-b_imuno * (t - tau))

    tumor_volume -= treatment_effect * tumor

    return tumor_volume


@jit(nopython=True)
def model_CumulativeDose(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    cumulative_dose = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            cumulative_dose += dose
            treatment_effect += a_imuno * cumulative_dose * np.exp(-b_imuno * (t - tau))

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_AccelED(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            treatment_effect += a_imuno * dose * np.exp(-b_imuno * (t - tau)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_AccelCD(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    cumulative_dose = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            cumulative_dose += dose
            treatment_effect += a_imuno * cumulative_dose * np.exp(-b_imuno * (t - tau)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


# Michaelis-Menten functional response
@jit(nopython=True)
def model_MichaelisMenten(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            treatment_effect += (a_imuno * dose) / (1 + b_imuno * dose)

    tumor_volume -= treatment_effect * tumor

    return tumor_volume


# de Pillis-Radunskaya law (rational form): d D^lam / (s T^lam + D^lam) per administration
@jit(nopython=True)
def model_dePillisRadunskayaLaw(y, t, growth_rate, carrying_capacity, d, s, lam, tau_immuno, dose_schedule):
    tumor = max(y[0], 1e-12)  # avoids a zero denominator
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    for i in range(len(dose_schedule)):
        dose = dose_schedule[i]
        tau = tau_immuno[i]
        if t >= tau:
            effect = d * (dose ** lam) / (s * (tumor ** lam) + dose ** lam)
            treatment_effect += effect

    tumor_volume -= treatment_effect * tumor

    return tumor_volume


# -- ED and CD with onset factors (same forms as in the radiotherapy stage)

@jit(nopython=True)
def model_CDwithLogFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, steepness, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            logistic_factor = 1 / (1 + np.exp(-steepness * time_since_treatment))
            cumulative_dose += dose
            treatment_effect += a_immuno * cumulative_dose * logistic_factor * np.exp(-b_immuno * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_CDwithExpFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, rate, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            cumulative_dose += dose
            treatment_effect += a_immuno * cumulative_dose * exponential_factor * np.exp(-b_immuno * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_CDwithSinFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, frequency, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            cumulative_dose += dose
            treatment_effect += a_immuno * cumulative_dose * sinusoidal_factor * np.exp(-b_immuno * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_EDwithLogFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, steepness, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            logistic_factor = 1 / (1 + np.exp(-steepness * time_since_treatment))
            treatment_effect += a_immuno * dose * logistic_factor * np.exp(-b_immuno * time_since_treatment)
    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_EDwithExpFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, rate, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            treatment_effect += a_immuno * dose * exponential_factor * np.exp(-b_immuno * time_since_treatment)
    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_EDwithSinFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, frequency, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            treatment_effect += a_immuno * dose * sinusoidal_factor * np.exp(-b_immuno * time_since_treatment)
    tumor_volume -= treatment_effect * tumor
    return tumor_volume


ALL_MODELS = [
    '_ExponentialDecay', '_CumulativeDose', '_AccelED', '_AccelCD',
    '_MichaelisMenten', '_dePillisRadunskayaLaw',
    '_CDwithLogFactor', '_CDwithExpFactor', '_CDwithSinFactor',
    '_EDwithLogFactor', '_EDwithExpFactor', '_EDwithSinFactor',
]

# Model with the lowest BIC (thesis, Section 4.4)
WINNER_MODEL = '_dePillisRadunskayaLaw'


def _treatment_param_names(model_extension):
    """Treatment parameters (between carrying_capacity and tau/dose_schedule)."""
    fn = globals()['model' + model_extension]
    fn = getattr(fn, 'py_func', fn)
    return list(_inspect.signature(fn).parameters)[4:-2]


def n_treatment_params(model_extension):
    return len(_treatment_param_names(model_extension))


# === Right-hand side: the numba-compiled function rather than .py_func ===
# .py_func is about 3.6x slower per solve. The compiled function is probed once per model and the pure
# Python function is used only if numba rejects it (identical results).
_RHS_CACHE = {}


def _rhs(model_extension):
    fn = globals()['model' + model_extension]
    cached = _RHS_CACHE.get(model_extension)
    if cached is not None:
        return cached
    py = getattr(fn, 'py_func', fn)
    try:
        import numpy as _np
        n = len(_inspect.signature(py).parameters) - 4  # y, t, r, cc, treatment parameters, tau, doses
        extra = (0.1,) * (n - 2)
        fn(_np.array([100.0]), 0.0, 0.1, 1000.0, *extra,
           _np.array([0.0]), _np.array([1.0]))
        chosen = fn                      # compiled function works
    except Exception:
        chosen = py                      # pure Python fallback
    _RHS_CACHE[model_extension] = chosen
    return chosen


def solve_model(model_extension, time_array, parameters, initial_condition,
                treatment_days, dose=DOSE_DEFAULT, type_sol='data'):
    """Solves at the observed times (or on the 'smooth' grid, dt=0.1).
    parameters = (growth_rate, carrying_capacity, *treatment_params)."""
    rhs = _rhs(model_extension)
    tau = np.asarray(treatment_days, dtype=float)
    dose_schedule = np.full(len(tau), float(dose))

    if type_sol == 'smooth':
        bgn, end = round(time_array[0], 1), round(time_array[-1], 1)
        time_array = np.linspace(bgn, end, int((end - bgn) / 0.1) + 1)

    if SOLVER == 'compiled':
        sol = compiled_ode.solve(rhs, time_array, parameters, initial_condition, tau, dose_schedule)
        if np.all(np.isfinite(sol)):
            return np.column_stack((time_array, sol)) if type_sol == 'smooth' else sol
        # stiff case (explicit steps exhausted): fall back to odeint below

    if type_sol == 'smooth':
        sol = odeint(rhs, y0=[initial_condition], t=time_array,
                     args=(*parameters, tau, dose_schedule), mxstep=10000)
        return np.column_stack((time_array, sol))

    return odeint(rhs, y0=[initial_condition], t=time_array,
                  args=(*parameters, tau, dose_schedule), mxstep=10000)
