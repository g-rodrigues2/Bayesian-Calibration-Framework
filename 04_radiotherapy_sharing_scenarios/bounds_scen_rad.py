"""Prior bounds and labels of the multi-group theta of the radiotherapy sharing scenarios.

Layout of theta:
  1) radiotherapy parameters in the order of MODEL_PARAM_NAMES:
       global -> 1 entry;  group -> 1 per group (in the order of `groups`)
  2) initial conditions T0 per mouse, in blocks per group
  3) std (always global)
"""
import _path_setup  # noqa: F401

from bounds_rad import _RAD_BOUNDS
from scenarios_rad import MODEL_PARAM_NAMES


# Priors of the radiotherapy parameters in the sharing scenarios, as reported in the thesis
# (Table 4.1): alpha ~ U(0, 1), beta ~ U(0, 1.5), delay ~ U(0, 8) days. The per-model bounds of
# bounds_rad are narrower (beta < 0.1) and would truncate the posterior of beta in the resistant group.
SCENARIO_BOUNDS = {
    '_ExponentialDecayDelay': {'alpha_rad': (0.0, 1.0), 'beta_rad': (0.0, 1.5), 'delay': (0.0, 8.0)},
}


def _group_tag(group_name):
    return group_name.replace('radiation_', '')


def define_scenario_bounds_labels(model_extension, scenario, groups, full_data,
                                  max_ic):
    """(l_bound, u_bound, labels) of the multi-group theta of a scenario."""
    param_names = MODEL_PARAM_NAMES[model_extension]
    labels_base, l_base, u_base = _RAD_BOUNDS[model_extension]
    bound_of = {p: (l_base[i], u_base[i]) for i, p in enumerate(param_names)}
    bound_of.update(SCENARIO_BOUNDS.get(model_extension, {}))

    l_bound, u_bound, labels = [], [], []
    for p in param_names:
        lo, hi = bound_of[p]
        if scenario[p] == 'global':
            l_bound.append(lo); u_bound.append(hi); labels.append(f'{p}_global')
        else:
            for g in groups:
                l_bound.append(lo); u_bound.append(hi)
                labels.append(f'{p}_{_group_tag(g)}')

    ic_max = max_ic * 2
    for g in groups:
        gtag = _group_tag(g)
        for k in range(len(full_data[g])):
            l_bound.append(0.0); u_bound.append(ic_max)
            labels.append(f'ic_{gtag}_{k + 1}')

    l_bound.append(0.1); u_bound.append(1000.0); labels.append('std')
    return l_bound, u_bound, labels
