"""Radiotherapy models and ODE solution with a dose schedule.

Each model is the right-hand side of dV/dt = logistic growth - radiation effect(t) V, where growth
(r, cc) is inherited from the control groups (fixed; see growth_params.py) and the radiation effect
depends on the dose schedule:

    tau_radiation  = days of the fractions (thesis schedule: [0, 1, 2, 3, 4, 5])
    dose_schedule  = [dose] * n_fractions (2 Gy per fraction)

The 19 models differ in the functional form of the effect (exponential decay, cumulative dose,
accelerated decay, logistic/exponential/sinusoidal onset factors and a delay).

Solvers:
  solve_model            -- compiled Dormand-Prince solver restarted at the effect onsets (compiled_ode.py,
                            default) or odeint (CALIB_SOLVER=odeint), at the observed times (18 models).
  solve_model_lq         -- piecewise integration between doses, applying the linear-quadratic
                            surviving fraction exp(-a d - b d^2) as a jump at each dose day (_LQ only).
  solve_model_dispatch   -- selects the solver for the model.
"""
import os

import numpy as np
from numba import jit
from scipy.integrate import odeint

import compiled_ode  # noqa: E402  (01_control_growth, on sys.path through _path_setup)

SOLVER = os.environ.get('CALIB_SOLVER', 'compiled').lower()
DOSE_DEFAULT = float(os.environ.get('CALIB_DOSE', '2'))  # Gy per fraction

# Model with a discrete effect (linear-quadratic surviving fraction)
DISCRETE_KILL_MODELS = {'_LQ'}
_AFTER_DOSE = 1e-9  # the post-dose volume is placed just after the dose day


@jit(nopython=True)
def model_LQ(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, tau_radiation, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    return tumor_volume


@jit(nopython=True)
def model_ExponentialDecay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, tau_radiation, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            treatment_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau))

    tumor_volume -= treatment_effect * tumor

    return tumor_volume

@jit(nopython=True)
def model_CumulativeDose(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * np.exp(-b_radiation * (t - tau))

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_AccelED(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            treatment_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_AccelCD(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * np.exp(-b_radiation * (t - tau)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_EDwithLogFactor(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, steepness, tau_radiation, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            time_since_treatment = t - tau
            logistic_factor = 1 / (1 + np.exp(-steepness * time_since_treatment))
            treatment_effect += a_radiation * dose * logistic_factor * np.exp(-b_radiation * time_since_treatment)
    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_CDwithLogFactor(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, steepness, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            time_since_treatment = t - tau
            logistic_factor = 1 / (1 + np.exp(-steepness * time_since_treatment))
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * logistic_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_EDwithExpFactor(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, rate, tau_radiation, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            time_since_treatment = t - tau
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            treatment_effect += a_radiation * dose *  exponential_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_CDwithExpFactor(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, rate, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            time_since_treatment = t - tau
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * exponential_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_EDwithSinFactor(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, frequency, tau_radiation, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            time_since_treatment = t - tau
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            treatment_effect += a_radiation * dose *  sinusoidal_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_CDwithSinFactor(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, frequency, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:
            time_since_treatment = t - tau
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * sinusoidal_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_ExponentialDecayDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, tau_radiation, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            treatment_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau - delay))

    tumor_volume -= treatment_effect * tumor

    return tumor_volume

@jit(nopython=True)
def model_CumulativeDoseDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * np.exp(-b_radiation * (t - tau - delay))

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_ACDDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * np.exp(-b_radiation * (t - tau - delay)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_AEDDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            treatment_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau - delay)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_CDEFDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, rate, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            time_since_treatment = t - tau - delay
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * exponential_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_CDSFDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, frequency, delay, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            time_since_treatment = t - tau - delay
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            cumulative_dose += dose
            treatment_effect += a_radiation * cumulative_dose * sinusoidal_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_EDSFDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, frequency, delay, tau_radiation, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            time_since_treatment = t - tau - delay
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            treatment_effect += a_radiation * dose *  sinusoidal_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume



@jit(nopython=True)
def model_EDEFDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, rate, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            time_since_treatment = t - tau - delay
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            treatment_effect += a_radiation * dose * exponential_factor * np.exp(-b_radiation * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


# === Model metadata: number of radiotherapy parameters (besides r and cc) ===
# Used by bounds_rad and inference; inferred from the signature (arguments between carrying_capacity and
# tau_radiation). Note: the order of `delay` and `rate` in model_EDEFDelay and model_CDEFDelay is
# (delay, rate), while their labels in bounds_rad list (rate, delay).
import inspect as _inspect


def _radiation_param_names(model_extension):
    """Names of the radiotherapy parameters of a model (between cc and tau_radiation)."""
    fn = globals()['model' + model_extension]
    fn = getattr(fn, 'py_func', fn)
    params = list(_inspect.signature(fn).parameters)
    # params = [y, t, growth_rate, carrying_capacity, <rad...>, tau_radiation, dose_schedule]
    return params[4:-2]


def n_radiation_params(model_extension):
    return len(_radiation_param_names(model_extension))


ALL_MODELS = [
    '_LQ', '_ExponentialDecay', '_CumulativeDose', '_AccelED', '_AccelCD',
    '_CDwithLogFactor', '_CDwithExpFactor', '_CDwithSinFactor',
    '_EDwithLogFactor', '_EDwithExpFactor', '_EDwithSinFactor',
    '_ExponentialDecayDelay', '_EDEFDelay', '_CumulativeDoseDelay', '_CDEFDelay',
    '_ACDDelay', '_AEDDelay', '_CDSFDelay', '_EDSFDelay',
]

# Selected model (used in the parameter-sharing scenarios).
WINNER_MODEL = '_ExponentialDecayDelay'


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
        n = len(_inspect.signature(py).parameters) - 4  # y, t, r, cc, radiotherapy parameters, tau, dose schedule
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
    """Solves a model with a continuous effect at the observed times (or on the 'smooth' grid).

    parameters = (growth_rate, carrying_capacity, *radiation_params).
    treatment_days = dose days; dose_schedule = [dose] * len(treatment_days).
    """
    rhs = _rhs(model_extension)
    tau_radiation = np.asarray(treatment_days, dtype=float)
    dose_schedule = np.full(len(tau_radiation), float(dose))

    if type_sol == 'smooth':
        bgn, end = round(time_array[0], 1), round(time_array[-1], 1)
        time_array = np.linspace(bgn, end, int((end - bgn) / 0.1) + 1)

    if SOLVER == 'compiled':
        names = _radiation_param_names(model_extension)
        delay = float(parameters[2 + names.index('delay')]) if model_extension.endswith('Delay') else 0.0
        sol = compiled_ode.solve(rhs, time_array, parameters, initial_condition, tau_radiation, dose_schedule,
                                 delay)
        if np.all(np.isfinite(sol)):
            return np.column_stack((time_array, sol)) if type_sol == 'smooth' else sol
        # stiff case (explicit steps exhausted): fall back to odeint below

    if type_sol == 'smooth':
        sol = odeint(rhs, y0=[initial_condition], t=time_array,
                     args=(*parameters, tau_radiation, dose_schedule), mxstep=2000)
        return np.column_stack((time_array, sol))

    return odeint(rhs, y0=[initial_condition], t=time_array,
                  args=(*parameters, tau_radiation, dose_schedule), mxstep=2000)


def solve_model_lq(model_extension, time_array, parameters, initial_condition,
                   treatment_days, dose=DOSE_DEFAULT, type_sol='data'):
    """Piecewise solution of the linear-quadratic model _LQ.

    The right-hand side of _LQ has no continuous treatment term: radiation acts as a discrete surviving
    fraction S = exp(-alpha d - beta d^2), applied as a multiplicative jump of the volume at each dose
    day, including a dose given at the initial time. Growth is integrated between doses and the solution
    is interpolated at the requested times. A volume measured on a dose day is the volume before that
    fraction (measurement before irradiation), as in the calibrations reported in the thesis.

    parameters = (growth_rate, carrying_capacity, alpha_rad, beta_rad).
    """
    rhs = _rhs(model_extension)
    alpha, beta = parameters[2], parameters[3]
    tau = np.asarray(treatment_days, dtype=float)
    dose_schedule = np.full(len(tau), float(dose))
    survival = np.exp(-alpha * dose - beta * dose ** 2)

    t0, t_end = float(time_array[0]), float(time_array[-1])
    doses = [float(x) for x in tau if t0 <= x <= t_end]

    ts, vs = [t0], [float(initial_condition)]
    n = float(initial_condition)
    cur = t0
    for stop in sorted(set(doses) | {t_end}):
        if stop > cur:
            seg = np.round(np.arange(cur, stop + 0.05, 0.1), 4)
            if seg[-1] < stop:
                seg = np.append(seg, stop)
            if SOLVER == 'compiled':
                # no continuous effect in _LQ: one smooth segment between doses
                sol = compiled_ode.solve(rhs, seg, parameters, n, tau[:0], dose_schedule[:0])
            if SOLVER != 'compiled' or not np.all(np.isfinite(sol)):
                sol = odeint(rhs, y0=[n], t=seg,
                             args=(*parameters, tau, dose_schedule), mxstep=2000)
            ts.extend(seg[1:].tolist()); vs.extend(sol[1:, 0].tolist())
            n = sol[-1, 0]
        if stop in doses:
            n *= survival ** doses.count(stop)
            ts.append(stop + _AFTER_DOSE); vs.append(n)   # jump just after the dose-day measurement
        cur = stop

    ts = np.asarray(ts); vs = np.asarray(vs)
    order = np.argsort(ts, kind='stable')
    ts, vs = ts[order], vs[order]

    if type_sol == 'smooth':
        out_t = np.linspace(t0, t_end, int((t_end - t0) / 0.1) + 1)
        return np.column_stack((out_t, np.interp(out_t, ts, vs)))
    # 'data' mode: same convention as the continuous solver (volume column only)
    out_t = np.asarray(time_array, dtype=float)
    return np.interp(out_t, ts, vs).reshape(-1, 1)


def solve_model_dispatch(model_extension, time_array, parameters,
                         initial_condition, treatment_days, dose=DOSE_DEFAULT,
                         type_sol='data'):
    """Selects the solver: piecewise for _LQ, continuous otherwise."""
    if model_extension in DISCRETE_KILL_MODELS:
        return solve_model_lq(model_extension, time_array, parameters,
                              initial_condition, treatment_days, dose, type_sol)
    return solve_model(model_extension, time_array, parameters,
                       initial_condition, treatment_days, dose, type_sol)
