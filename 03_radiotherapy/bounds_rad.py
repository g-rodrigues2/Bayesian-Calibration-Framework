"""Prior bounds (uniform priors) and labels of the radiotherapy parameters of each model.

The growth parameters (r, cc) are not calibrated here: they are inherited from the control groups and
fixed (growth_params.py). Each model calibrates its radiotherapy parameters (alpha_rad, beta_rad and,
depending on the model, delay/rate/steepness/frequency), one initial condition per mouse and one std.

These are the bounds of the model-selection runs reported in the thesis. The group calibrations used
different upper bounds for the two phenotypes: `_RAD_BOUNDS` holds the radiation-sensitive values and
`_RESISTANT_UPPER` the radiation-resistant upper bounds; the mouse-specific calibrations used the wider
`_MOUSE_SPECIFIC_UPPER` (all lower bounds are zero).
"""
from radiation_models import n_radiation_params

# Bounds of the radiotherapy parameters per model: (labels, lower, upper).
_RAD_BOUNDS = {
    '_LQ':                 (["alpha_rad", "beta_rad"],            [0.0, 0.0],       [0.5, 0.1]),
    '_ExponentialDecay':   (["alpha_rad", "beta_rad"],            [0.0, 0.0],       [0.1, 0.1]),
    '_CumulativeDose':     (["alpha_rad", "beta_rad"],            [0.0, 0.0],       [0.1, 0.1]),
    '_AccelED':            (["alpha_rad", "beta_rad"],            [0.0, 0.0],       [0.1, 0.1]),
    '_AccelCD':            (["alpha_rad", "beta_rad"],            [0.0, 0.0],       [0.1, 0.1]),
    '_CDwithLogFactor':    (["alpha_rad", "beta_rad", "steepness"], [0.0, 0.0, 0.0], [0.1, 0.1, 0.5]),
    '_CDwithExpFactor':    (["alpha_rad", "beta_rad", "rate"],    [0.0, 0.0, 0.0],  [0.1, 0.5, 0.1]),
    '_CDwithSinFactor':    (["alpha_rad", "beta_rad", "frequency"], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
    '_EDwithLogFactor':    (["alpha_rad", "beta_rad", "steepness"], [0.0, 0.0, 0.0], [0.1, 0.1, 1.0]),
    '_EDwithExpFactor':    (["alpha_rad", "beta_rad", "rate"],    [0.0, 0.0, 0.0],  [0.1, 0.1, 1.0]),
    '_EDwithSinFactor':    (["alpha_rad", "beta_rad", "frequency"], [0.0, 0.0, 0.0], [0.1, 1.0, 2.0]),
    '_ExponentialDecayDelay': (["alpha_rad", "beta_rad", "delay"], [0.0, 0.0, 0.0], [0.1, 0.1, 8.0]),
    '_EDEFDelay':          (["alpha_rad", "beta_rad", "rate", "delay"], [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 1.0, 8.0]),
    '_CumulativeDoseDelay': (["alpha_rad", "beta_rad", "delay"], [0.0, 0.0, 0.0],  [0.1, 0.1, 8.0]),
    '_CDEFDelay':          (["alpha_rad", "beta_rad", "rate", "delay"], [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 1.0, 8.0]),
    '_ACDDelay':           (["alpha_rad", "beta_rad", "delay"], [0.0, 0.0, 0.0],   [0.1, 0.1, 8.0]),
    '_AEDDelay':           (["alpha_rad", "beta_rad", "delay"], [0.0, 0.0, 0.0],   [0.15, 0.1, 8.0]),
    '_CDSFDelay':          (["alpha_rad", "beta_rad", "frequency", "delay"], [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.2, 8.0]),
    '_EDSFDelay':          (["alpha_rad", "beta_rad", "frequency", "delay"], [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 5.0, 8.0]),
}


# Upper bounds of the radiation-resistant group (same labels and lower bounds as _RAD_BOUNDS).
_RESISTANT_UPPER = {
    '_LQ':                    [0.1, 0.1],
    '_ExponentialDecay':      [0.1, 1.0],
    '_CumulativeDose':        [0.1, 1.0],
    '_AccelED':               [0.1, 0.1],
    '_AccelCD':               [0.1, 2.0],
    '_CDwithLogFactor':       [0.1, 2.0, 0.1],
    '_CDwithExpFactor':       [0.1, 1.0, 1.0],
    '_CDwithSinFactor':       [0.1, 1.0, 2.0],
    '_EDwithLogFactor':       [0.1, 1.0, 0.1],
    '_EDwithExpFactor':       [0.1, 1.0, 1.0],
    '_EDwithSinFactor':       [0.1, 1.0, 1.0],
    '_ExponentialDecayDelay': [0.1, 1.5, 8.0],
    '_EDEFDelay':             [0.1, 1.5, 1.0, 8.0],
    '_CumulativeDoseDelay':   [0.1, 1.5, 8.0],
    '_CDEFDelay':             [0.1, 1.5, 1.0, 8.0],
    '_ACDDelay':              [0.1, 1.5, 8.0],
    '_AEDDelay':              [0.1, 1.5, 8.0],
    '_CDSFDelay':             [0.1, 1.5, 3.0, 8.0],
    '_EDSFDelay':             [0.1, 1.5, 2.0, 8.0],
}


# Upper bounds of the mouse-specific calibrations reported in the thesis (Tables S2.2 and S2.3). These runs
# used wider boxes than the group calibrations: 1 for every radiotherapy parameter of the models without
# delay, and 2 for alpha, beta and the onset parameter of the delayed models, with the delay up to 8 days.
# In _EDEFDelay and _CDEFDelay the original runs listed (rate, delay) with upper bounds (2, 8) while the
# model reads (delay, rate), so the delay was bounded by 2 and the rate by 8; the same positions are kept.
_MOUSE_SPECIFIC_UPPER = {
    **{m: [1.0] * len(_RAD_BOUNDS[m][0]) for m in _RAD_BOUNDS if not m.endswith('Delay')},
    **{m: [2.0] * (len(_RAD_BOUNDS[m][0]) - 1) + [8.0] for m in _RAD_BOUNDS if m.endswith('Delay')},
}


def find_max_initial_condition(full_data):
    """Largest T(0) observed in any group."""
    largest = -float('inf')
    for mice in full_data.values():
        for m in mice:
            largest = max(largest, m['data'][0, 1])
    return largest


def define_bounds_labels(model_extension, group_size, max_ic,
                         bool_multiple_mice=False, group=None):
    """(l_bound, u_bound, labels) of the radiotherapy parameters, initial conditions and std.

    Order of theta: [radiotherapy parameters..., ic (1..N or 1), std]. In the group mode,
    group='radiation_resistant' uses the upper bounds of the resistant group and any other value the
    sensitive ones; the mouse-specific mode uses the wider bounds of `_MOUSE_SPECIFIC_UPPER` for both groups.
    """
    if model_extension not in _RAD_BOUNDS:
        raise ValueError(f"Unknown radiotherapy model: {model_extension}")
    labels, l_bound, u_bound = _RAD_BOUNDS[model_extension]
    labels, l_bound, u_bound = list(labels), list(l_bound), list(u_bound)
    if not bool_multiple_mice:
        u_bound = list(_MOUSE_SPECIFIC_UPPER[model_extension])
    elif group == 'radiation_resistant':
        u_bound = list(_RESISTANT_UPPER[model_extension])
    assert len(labels) == n_radiation_params(model_extension)

    ic_max = max_ic * 2
    n_ic = group_size if bool_multiple_mice else 1
    for i in range(n_ic):
        labels.append(f"ic{i + 1}" if bool_multiple_mice else "ic")
        l_bound.append(0.0)
        u_bound.append(ic_max)

    labels.append("std")
    l_bound.append(0.1)
    u_bound.append(1000.0)
    return l_bound, u_bound, labels
