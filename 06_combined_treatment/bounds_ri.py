"""Uniform priors. theta = [model parameters, ic_1..ic_N, std]."""
from rad_immuno_models import MODEL_PARAMS

PARAM_BOUNDS = {
    '_RadImmunoFixed': ([], []),
    '_RadImmunoFitD': ([0.01], [2.0]),                  # prior of d in Table 4.15 of the thesis
    '_SUVLinear': ([-10.0, -10.0], [10.0, 10.0]),
    '_SUVMichaelisMenten': ([-2.0, -2.0], [3.0, 3.0]),
    '_SUVLogistic': ([-2.0, -2.0], [3.0, 3.0]),
    '_SUVSatExp': ([-2.0, -2.0], [3.0, 3.0]),
}
IC_UPPER_FACTOR = 2.0                                   # ic in (0, 2 x largest initial volume)
STD_BOUNDS = {'prediction': (0.1, 1000.0), 'suv': (0.1, 300.0), 'loo': (0.1, 300.0)}


def define_bounds_labels(model, n_mice, max_ic, analysis):
    l_bound, u_bound = map(list, PARAM_BOUNDS[model])
    labels = list(MODEL_PARAMS[model])
    for i in range(n_mice):
        labels.append(f'ic{i + 1}')
        l_bound.append(0.0)
        u_bound.append(IC_UPPER_FACTOR * max_ic)
    labels.append('std')
    l_bound.append(STD_BOUNDS[analysis][0])
    u_bound.append(STD_BOUNDS[analysis][1])
    return l_bound, u_bound, labels
