"""Log-likelihood, prior bounds and number of observations of each BIC cell, built from the code of the
stage that produced it (same data, fixed upstream parameters and priors as the calibrations).
"""
from functools import lru_cache

import _stage_paths  # noqa: F401
import numpy as np


@lru_cache(maxsize=None)
def _control():
    from data_loader import load_control_data
    from bounds import find_max_initial_condition
    data = load_control_data()
    return data, find_max_initial_condition(data)[0]


@lru_cache(maxsize=None)
def _radiation():
    from bounds_rad import find_max_initial_condition
    from data_loader_rad import load_radiation_data
    from growth_params import _fixed_for
    data = load_radiation_data()
    return data, find_max_initial_condition(data), _fixed_for(list(data))


@lru_cache(maxsize=None)
def _immuno():
    from bounds_imm import find_max_initial_condition
    from data_loader_imm import load_immuno_data
    from growth_params import _fixed_for
    data = load_immuno_data()
    return data, find_max_initial_condition(data), _fixed_for(list(data))


@lru_cache(maxsize=None)
def _combined():
    from data_loader_ri import load_data, max_initial_volume
    from thesis_params import upstream_params
    mice = load_data()['rad+io_sensitive']
    return mice, max_initial_volume(mice), upstream_params('sensitive')


def _mouse_index(mice, name):
    return [m['name'] for m in mice].index(name)


def problem(cell):
    """(loglik(theta), l_bound, u_bound, labels, n_obs) of a cell."""
    kind, model, block, row = cell['kind'], cell['model'], cell['block'], cell['row']

    if kind == 'control':
        from bounds import define_bounds_labels
        from inference import log_likelihood
        data, max_ic = _control()
        idx = None if row == 'Global' else _mouse_index(data[block], row)
        lb, ub, labels = define_bounds_labels(model, len(data[block]), max_ic, idx is None)
        mice = data[block] if idx is None else [data[block][idx]]
        return (lambda th: log_likelihood(th, model, data, block, idx)), lb, ub, labels, _n(mice)

    if kind == 'control_sharing':
        from bounds_shared import define_shared_bounds_labels
        from inference_shared import log_likelihood_shared
        from scenarios import SCENARIOS
        data, max_ic = _control()
        groups = ['control_sensitive', 'control_resistant']
        scenario = SCENARIOS[model][cell['scenario']]
        lb, ub, labels = define_shared_bounds_labels(model, scenario, groups, data, max_ic)
        n = sum(_n(data[g]) for g in groups)
        return (lambda th: log_likelihood_shared(th, model, data, groups, scenario)), lb, ub, labels, n

    if kind == 'radiation':
        from bounds_rad import define_bounds_labels
        from inference_rad import log_likelihood
        data, max_ic, growth = _radiation()
        idx = None if row == 'Global' else _mouse_index(data[block], row)
        lb, ub, labels = define_bounds_labels(model, len(data[block]), max_ic, idx is None, group=block)
        mice = data[block] if idx is None else [data[block][idx]]
        return (lambda th: log_likelihood(th, model, data, block, growth, idx)), lb, ub, labels, _n(mice)

    if kind == 'radiation_sharing':
        from bounds_scen_rad import define_scenario_bounds_labels
        from inference_scen_rad import log_likelihood_shared
        from scenarios_rad import SCENARIOS
        data, max_ic, growth = _radiation()
        groups = ['radiation_sensitive', 'radiation_resistant']
        scenario = SCENARIOS[model][cell['scenario']]
        lb, ub, labels = define_scenario_bounds_labels(model, scenario, groups, data, max_ic)
        n = sum(_n(data[g]) for g in groups)
        return (lambda th: log_likelihood_shared(th, model, data, groups, scenario, growth)), lb, ub, labels, n

    if kind == 'immuno':
        from bounds_imm import define_bounds_labels
        from inference_imm import log_likelihood
        data, max_ic, growth = _immuno()
        idx = None if row == 'Global' else _mouse_index(data[block], row)
        lb, ub, labels = define_bounds_labels(model, len(data[block]), max_ic, idx is None)
        mice = data[block] if idx is None else [data[block][idx]]
        return (lambda th: log_likelihood(th, model, data, block, growth, idx)), lb, ub, labels, _n(mice)

    if kind == 'suv':
        from bounds_ri import define_bounds_labels
        from inference_ri import log_likelihood
        mice, max_ic, upstream = _combined()
        lb, ub, labels = define_bounds_labels(model, len(mice), max_ic, 'suv')
        return (lambda th: log_likelihood(th, model, mice, upstream)), lb, ub, labels, _n(mice)

    raise ValueError(f'unknown kind {kind!r}')


def _n(mice):
    return int(sum(m['data'].shape[0] for m in mice))


def safe(loglik):
    """Wraps a log-likelihood so that exceptions and non-finite values return -inf."""
    def f(theta):
        try:
            value = float(loglik(np.asarray(theta, dtype=float)))
        except Exception:
            return -np.inf
        return value if np.isfinite(value) else -np.inf
    return f
