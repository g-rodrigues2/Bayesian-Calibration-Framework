"""Uniform priors (bounds) and parameter labels of each growth model.

The initial-condition upper bound (``ic_max``) is twice the largest observed T(0) over all mice
(``find_max_initial_condition()``), so that the free initial condition of each mouse covers the scale
of the data without being excessively wide.

The logistic bounds are shared by the global and mouse-specific modes: r in [0, 1] and cc in
[100, 20000].
"""


def find_max_initial_condition(full_data):
    """Returns (largest observed T0, its group, its mouse name)."""
    largest_ic = -float('inf')
    group_max_ic = None
    mouse_max_ic = None
    for group, mice in full_data.items():
        for mouse in mice:
            ic = mouse['data'][0, 1]
            if ic > largest_ic:
                largest_ic = ic
                group_max_ic = group
                mouse_max_ic = mouse['name']
    return largest_ic, group_max_ic, mouse_max_ic


def define_bounds_labels(model_extension, group_size, max_ic,
                         bool_multiple_mice=False):
    """Returns (l_bound, u_bound, labels) for the model.

    Parameters:
      model_extension     - model suffix (e.g. '_exp', '_log')
      group_size          - number of mice in the group (initial conditions)
      max_ic              - largest observed T(0) over all groups
      bool_multiple_mice  - True in the global mode: adds ic1..ic{group_size} and std;
                            False in the mouse-specific mode: adds 'ic' and 'std'.
    """
    ic_max = max_ic * 2

    if model_extension == '_exp':
        labels  = ["r"]
        l_bound = [0.0]
        u_bound = [1.0]
    elif model_extension == '_mendel':
        labels  = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [1.0,  1.0]
    elif model_extension == '_bert':
        labels  = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [2.0,  1.0]
    elif model_extension == '_lin':
        labels  = ["r", "b"]
        l_bound = [0.0, 0.0]
        u_bound = [1.0, ic_max]
    elif model_extension == '_gomp':
        labels  = ["r", "b",         "c"]
        l_bound = [1e-6, 1e-6,       1e-6]
        u_bound = [1.0,  10*ic_max,  ic_max]
    elif model_extension == '_surf':
        labels  = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [1.0,  ic_max]
    elif model_extension == '_log':
        labels  = ["r", "cc"]
        l_bound = [0.0,  100.]
        u_bound = [1.0,  20000]
    else:
        raise ValueError(f"Unknown model: {model_extension}")

    if bool_multiple_mice:
        for i in range(group_size):
            labels.append("ic" + str(i + 1))
            l_bound.append(0.0)
            u_bound.append(ic_max)
    else:
        labels.append("ic")
        l_bound.append(0.0)
        u_bound.append(ic_max)

    labels.append("std")
    l_bound.append(0.1)
    u_bound.append(1000.0)

    return l_bound, u_bound, labels
