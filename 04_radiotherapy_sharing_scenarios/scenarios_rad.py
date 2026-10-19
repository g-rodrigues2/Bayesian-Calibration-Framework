"""The 8 scenarios (V1-V8) for sharing the radiotherapy parameters (alpha_rad, beta_rad, delay) between
radiation_sensitive and radiation_resistant, for the selected delayed exponential decay model.

Each scenario marks each parameter as 'global' (one value for both groups) or 'group' (one value per
group). The generic partition of theta of `scenarios.py` (02_control_sharing_scenarios) is reused.
The exact definitions are in SCENARIOS (Table 4.3 of the thesis).
"""
import _path_setup  # noqa: F401  (adds scenarios.py to sys.path)

from scenarios import (  # noqa: F401  (generic partition of theta, re-exported for this stage)
    count_model_params, ic_slice_for_group, split_model_args_for_group,
)

WINNER_MODEL = '_ExponentialDecayDelay'

MODEL_PARAM_NAMES = {
    WINNER_MODEL: ['alpha_rad', 'beta_rad', 'delay'],
}

# Scenarios (alpha, beta, delay -> 'global' | 'group')
SCENARIOS = {
    WINNER_MODEL: {
        1: {'alpha_rad': 'group',  'beta_rad': 'group',  'delay': 'group'},
        2: {'alpha_rad': 'global', 'beta_rad': 'group',  'delay': 'global'},
        3: {'alpha_rad': 'group',  'beta_rad': 'global', 'delay': 'global'},
        4: {'alpha_rad': 'global', 'beta_rad': 'global', 'delay': 'global'},
        5: {'alpha_rad': 'group',  'beta_rad': 'group',  'delay': 'global'},
        6: {'alpha_rad': 'group',  'beta_rad': 'global', 'delay': 'group'},
        7: {'alpha_rad': 'global', 'beta_rad': 'group',  'delay': 'group'},
        8: {'alpha_rad': 'global', 'beta_rad': 'global', 'delay': 'group'},
    },
}


def scenario_tag(n):
    return f'V{int(n)}'


def scenario_id(n):
    return f'_V{int(n)}'


def describe(model_extension, n):
    """Text description of the sharing in a scenario."""
    sdef = SCENARIOS[model_extension][n]
    return ', '.join(
        f"{p}={'shared' if k == 'global' else 'group-specific'}"
        for p, k in sdef.items())
