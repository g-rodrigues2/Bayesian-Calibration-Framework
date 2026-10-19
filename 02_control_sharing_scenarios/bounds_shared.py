"""Prior bounds and labels of the multi-group theta, with parameter sharing given by
`scenarios.SCENARIOS`. The bounds of the model parameters come from `bounds.define_bounds_labels`.
"""
import _path_setup  # adds 01_control_growth/ to sys.path  # noqa: F401

from bounds import define_bounds_labels
from scenarios import MODEL_PARAM_NAMES


def _group_tag(group_name):
    """control_sensitive -> sensitive."""
    return group_name.replace('control_', '')


def define_shared_bounds_labels(model_extension, scenario, groups, full_data,
                                max_ic):
    """(l_bound, u_bound, labels) of the multi-group theta.

    Arguments:
      model_extension - '_exp' or '_log'
      scenario        - dict {param: 'global'|'group'} (from scenarios.SCENARIOS)
      groups          - group names (e.g. ['control_sensitive', 'control_resistant'])
      full_data       - dict {group: [mice...]} (used to count the mice of each group)
      max_ic          - largest T(0) observed in any group

    Layout: see the docstring of `scenarios.py`. The three lists follow the order of theta.
    """
    param_names = MODEL_PARAM_NAMES[model_extension]

    # Bounds of the model parameters from the single-mouse layout
    # [param1, ..., paramN, ic, std]; only the first N entries are used.
    base_l, base_u, _ = define_bounds_labels(
        model_extension, group_size=1, max_ic=max_ic,
        bool_multiple_mice=False,
    )
    n_model = len(param_names)
    model_l = base_l[:n_model]
    model_u = base_u[:n_model]

    l_bound, u_bound, labels = [], [], []

    # 1) Model parameters, in the order of param_names
    for i, p in enumerate(param_names):
        kind = scenario[p]
        if kind == 'global':
            l_bound.append(model_l[i])
            u_bound.append(model_u[i])
            labels.append(f'{p}_global')
        elif kind == 'group':
            for g in groups:
                l_bound.append(model_l[i])
                u_bound.append(model_u[i])
                labels.append(f'{p}_{_group_tag(g)}')
        else:
            raise ValueError(
                f'invalid kind for {p}: {kind!r} (expected "global" or "group")'
            )

    # 2) Initial conditions T0 per mouse, in blocks per group
    ic_max = max_ic * 2
    for g in groups:
        gtag = _group_tag(g)
        for k in range(len(full_data[g])):
            l_bound.append(0.0)
            u_bound.append(ic_max)
            labels.append(f'ic_{gtag}_{k + 1}')

    # 3) Global std
    l_bound.append(0.1)
    u_bound.append(1000.0)
    labels.append('std')

    return l_bound, u_bound, labels
