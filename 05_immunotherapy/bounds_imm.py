"""Prior bounds (uniform priors) and labels of the immunotherapy parameters.

Growth (r, cc) is not calibrated here: it is inherited from the control groups and fixed.

The upper bounds of beta_immuno of _CumulativeDose and _AccelCD were widened after the thesis defense
(see below); all other values are those of the original calibration.

Mixed parameterization: `ms_params` lists the treatment parameters calibrated per mouse (one theta
entry per mouse); the others are global. E.g. ms_params=['d'] -> d1..dN per mouse, others global.

Layout of theta:
  [<parameters (global = 1 entry, per mouse = N entries each)>, ic_1..ic_N, std]
"""
from immuno_models import ALL_MODELS as _ALL, _treatment_param_names

# (labels, l_bound, u_bound)
_TREAT_BOUNDS = {
    '_ExponentialDecay':      (['alpha_immuno', 'beta_immuno'], [0.01, 1.5], [1.0, 3.0]),
    # CD and ACD: upper bound of beta_immuno widened (2026-09-14). With the original bounds -- CD [5, 8]
    # and ACD [1.5, 5] -- the prior predictive check left observations above the 5-95% band (coverage of
    # the worst-covered mouse 84.2% and 63.2%): the cumulative dose multiplies the effect and the prior
    # produced too strong a tumor kill. alpha_immuno and the lower bounds are unchanged.
    #
    # Rerun with the compiled solver (2026-09-18, 3 seeds x 20000 draws, ppc_grid.py protocol): CD covers
    # every observation from 11 upward, so the adopted 15 is conservative. ACD at 100 still leaves one
    # observation of one mouse outside the band in 2 of the 3 sets; complete coverage holds between 150
    # and 300 and is lost again at 500 (a very wide box makes the small-mu draws that form the upper edge
    # of the band rare), so the criterion is not monotone in the bound and 100 was kept -- ACD is rejected
    # by a wide margin by the BIC anyway. The earlier choice of 100 rested on runs with odeint, which does
    # not restart at each dose: under odeint ACD already covered everything at 60.
    '_CumulativeDose':        (['alpha_immuno', 'beta_immuno'], [0.01, 5.0], [1.0, 15.0]),
    '_AccelED':               (['alpha_immuno', 'beta_immuno'], [0.01, 1.5], [1.0, 5.0]),
    '_AccelCD':               (['alpha_immuno', 'beta_immuno'], [0.01, 1.5], [1.0, 100.0]),
    '_MichaelisMenten':       (['alpha_immuno', 'beta_immuno'], [0.0, 20.0], [1.0, 60.0]),
    '_dePillisRadunskayaLaw': (['d', 's', 'lam'], [0.01, 0.01, 0.01], [2.0, 2.0, 2.0]),
    '_CDwithLogFactor':       (['alpha_immuno', 'beta_immuno', 'steepness'], [0.0, 2.0, 0.0], [1.0, 6.5, 1.0]),
    '_CDwithExpFactor':       (['alpha_immuno', 'beta_immuno', 'rate'], [0.0, 0.0, 0.0], [1.0, 2.5, 1.0]),
    '_CDwithSinFactor':       (['alpha_immuno', 'beta_immuno', 'frequency'], [0.0, 0.0, 0.0], [1.0, 7.0, 1.0]),
    '_EDwithLogFactor':       (['alpha_immuno', 'beta_immuno', 'steepness'], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]),
    '_EDwithExpFactor':       (['alpha_immuno', 'beta_immuno', 'rate'], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    '_EDwithSinFactor':       (['alpha_immuno', 'beta_immuno', 'frequency'], [0.0, 0.0, 0.0], [1.0, 3.0, 1.0]),
}

# Models calibrated by default (all models of immuno_models)
CALIBRATABLE_MODELS = [m for m in _ALL if m in _TREAT_BOUNDS]

# Initial-volume and std bounds of the group and mouse-specific calibrations (thesis Table S3.1)
IC_LOW_MULTI = 60.0
IC_LOW_SINGLE = 60.0
STD_BOUNDS = (0.0, 200.0)
STD_BOUNDS_SINGLE = (0.1, 200.0)


def find_max_initial_condition(full_data):
    largest = -float('inf')
    for mice in full_data.values():
        for m in mice:
            largest = max(largest, m['data'][0, 1])
    return largest


def treatment_param_names(model_extension):
    if model_extension not in _TREAT_BOUNDS:
        raise ValueError(
            f'Unknown immunotherapy model: {model_extension}. '
            f'Available models: {CALIBRATABLE_MODELS}.')
    return list(_TREAT_BOUNDS[model_extension][0])


def count_treatment_slots(model_extension, group_size, ms_params=()):
    ms = set(ms_params or ())
    return sum(group_size if p in ms else 1
               for p in treatment_param_names(model_extension))


def define_bounds_labels(model_extension, group_size, max_ic,
                         bool_multiple_mice=False, ms_params=()):
    """(l_bound, u_bound, labels).

    bool_multiple_mice=True  -> group mode: one initial condition per mouse; parameters in
                                `ms_params` expanded per mouse.
    bool_multiple_mice=False -> single-mouse mode: one initial condition; ms_params ignored.
    """
    if model_extension not in _TREAT_BOUNDS:
        raise ValueError(
            f'Unknown immunotherapy model: {model_extension}. '
            f'Available models: {CALIBRATABLE_MODELS}')
    base_labels, base_l, base_u = _TREAT_BOUNDS[model_extension]
    # the number of bounds must match the ODE signature
    assert len(base_labels) == len(_treatment_param_names(model_extension)), (
        f'{model_extension}: {len(base_labels)} bounds vs '
        f'{len(_treatment_param_names(model_extension))} ODE parameters')

    ic_max = max_ic * 2
    labels, l_bound, u_bound = [], [], []
    ms = set(ms_params or ()) if bool_multiple_mice else set()

    for lbl, lo, hi in zip(base_labels, base_l, base_u):
        if lbl in ms:
            for i in range(group_size):
                labels.append(f'{lbl}{i + 1}')
                l_bound.append(lo); u_bound.append(hi)
        else:
            labels.append(lbl)
            l_bound.append(lo); u_bound.append(hi)

    if bool_multiple_mice:
        for i in range(group_size):
            labels.append(f'ic{i + 1}')
            l_bound.append(IC_LOW_MULTI); u_bound.append(ic_max)
    else:
        labels.append('ic')
        l_bound.append(IC_LOW_SINGLE); u_bound.append(ic_max)

    labels.append('std')
    std_bounds = STD_BOUNDS if bool_multiple_mice else STD_BOUNDS_SINGLE
    l_bound.append(std_bounds[0]); u_bound.append(std_bounds[1])
    return l_bound, u_bound, labels


def extract_treatment_params(theta, model_extension, n_mice, mouse_idx,
                             ms_params=()):
    """(parameters of the mouse, n_slots). Parameters in `ms_params` take n_mice consecutive
    positions; the others take one."""
    ms = set(ms_params or ())
    params, pos = [], 0
    for lbl in treatment_param_names(model_extension):
        if lbl in ms:
            params.append(theta[pos + mouse_idx])
            pos += n_mice
        else:
            params.append(theta[pos])
            pos += 1
    return tuple(params), pos
