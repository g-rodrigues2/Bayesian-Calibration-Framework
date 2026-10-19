"""Scenarios for sharing the growth parameters between the control groups.

Each scenario marks each model parameter as
  - 'global' : a single value shared by all groups
  - 'group'  : one value per group

`std` is always global (one noise level for both groups).

Layout of theta:
  1) model parameters in the order of MODEL_PARAM_NAMES:
       - 'global' => 1 entry
       - 'group'  => len(groups) entries (in the order of `groups`)
  2) initial conditions T0 per mouse, in blocks per group:
       [T0_g0_m1, ..., T0_g0_mN0, T0_g1_m1, ..., T0_g1_mN1]
  3) std (1 entry)
"""

# Model parameters (in the order of the signature model_<ext>(y, t, ...)):
MODEL_PARAM_NAMES = {
    '_exp': ['r'],
    '_log': ['r', 'cc'],
}

# Valid scenarios per model. For _exp (no carrying capacity) only scenarios 1 (group) and 4 (global).
SCENARIOS = {
    '_exp': {
        1: {'r': 'group'},
        4: {'r': 'global'},
    },
    '_log': {
        1: {'r': 'group',  'cc': 'group'},
        2: {'r': 'group',  'cc': 'global'},
        3: {'r': 'global', 'cc': 'group'},
        4: {'r': 'global', 'cc': 'global'},
    },
}


def scenario_id(scenario_num):
    """File-name tag '_s<N>'."""
    return f'_s{scenario_num}'


def count_model_params(scenario, groups):
    """Number of theta entries used by the model parameters."""
    n = 0
    for p, kind in scenario.items():
        n += 1 if kind == 'global' else len(groups)
    return n


def split_model_args_for_group(theta, scenario, param_names, groups, group_name):
    """Tuple of model arguments of one group, in the order of `param_names`."""
    ode_args = []
    offset = 0
    for p in param_names:
        kind = scenario[p]
        if kind == 'global':
            ode_args.append(theta[offset])
            offset += 1
        else:
            gidx = groups.index(group_name)
            ode_args.append(theta[offset + gidx])
            offset += len(groups)
    return tuple(ode_args), offset


def ic_slice_for_group(theta, model_param_count, full_data, groups, group_name):
    """Initial conditions T0 of group `group_name`; `model_param_count` is the value returned by
    `count_model_params` (offset of the first initial condition)."""
    offset = model_param_count
    for g in groups:
        ndim_g = len(full_data[g])
        if g == group_name:
            return theta[offset:offset + ndim_g]
        offset += ndim_g
    raise KeyError(f'Unknown group: {group_name}')
