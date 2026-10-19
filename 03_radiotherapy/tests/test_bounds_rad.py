"""Tests of the radiotherapy bounds."""
import pytest

import radiation_models as rm
from bounds_rad import define_bounds_labels, find_max_initial_condition


@pytest.mark.parametrize('model', rm.ALL_MODELS)
def test_ndim_global(model):
    """ndim = n_rad + group_size (ICs) + 1 (std)."""
    l, u, lab = define_bounds_labels(model, group_size=9, max_ic=200.0,
                                     bool_multiple_mice=True)
    n_rad = rm.n_radiation_params(model)
    assert len(l) == len(u) == len(lab) == n_rad + 9 + 1
    assert lab[-1] == 'std'
    assert lab[n_rad] == 'ic1'


@pytest.mark.parametrize('model', rm.ALL_MODELS)
def test_ndim_mouse(model):
    l, u, lab = define_bounds_labels(model, group_size=9, max_ic=200.0,
                                     bool_multiple_mice=False)
    assert len(l) == rm.n_radiation_params(model) + 2
    assert lab[-2:] == ['ic', 'std']


@pytest.mark.parametrize('model', rm.ALL_MODELS)
def test_lower_below_upper(model):
    l, u, _ = define_bounds_labels(model, group_size=3, max_ic=200.0,
                                   bool_multiple_mice=True)
    assert all(lo < hi for lo, hi in zip(l, u))


def test_rad_params_come_first():
    l, u, lab = define_bounds_labels('_ExponentialDecayDelay', 5, 200.0,
                                     bool_multiple_mice=True)
    assert lab[:3] == ['alpha_rad', 'beta_rad', 'delay']
    assert (l[2], u[2]) == (0.0, 8.0)          # delay bound


def test_aeddelay_alpha_bound_is_special():
    """_AEDDelay has alpha up to 0.15 in the group calibration."""
    l, u, _ = define_bounds_labels('_AEDDelay', 1, 200.0, bool_multiple_mice=True)
    assert u[0] == 0.15


@pytest.mark.parametrize('model', rm.ALL_MODELS)
def test_mouse_specific_upper_bounds(model):
    """Mouse-specific runs: 1 for every parameter without delay; 2 and a delay up to 8 with delay."""
    _, u_s, _ = define_bounds_labels(model, 1, 200.0, False, group='radiation_sensitive')
    _, u_r, _ = define_bounds_labels(model, 1, 200.0, False, group='radiation_resistant')
    n_rad = rm.n_radiation_params(model)
    assert u_s[:n_rad] == u_r[:n_rad]
    if model.endswith('Delay'):
        assert u_s[:n_rad] == [2.0] * (n_rad - 1) + [8.0]
    else:
        assert u_s[:n_rad] == [1.0] * n_rad


def test_ic_and_std_bounds():
    l, u, lab = define_bounds_labels('_ExponentialDecay', 2, 150.0,
                                     bool_multiple_mice=True)
    ics = [i for i, x in enumerate(lab) if x.startswith('ic')]
    for i in ics:
        assert (l[i], u[i]) == (0.0, 300.0)     # 2*max_ic
    assert (l[-1], u[-1]) == (0.1, 1000.0)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match='Unknown radiotherapy model'):
        define_bounds_labels('_unknown', 1, 100.0)


def test_find_max_initial_condition(synthetic_rad_data):
    m = find_max_initial_condition(synthetic_rad_data)
    # largest T0 of the fixture: resistant c1_m1 = 110 * 1.5 = 165
    assert m == pytest.approx(165.0)


def test_resistant_group_upper_bounds():
    """The resistant group uses its own upper bounds; lower bounds and labels are shared."""
    l_s, u_s, lab_s = define_bounds_labels('_ExponentialDecayDelay', 2, 200.0, True,
                                           group='radiation_sensitive')
    l_r, u_r, lab_r = define_bounds_labels('_ExponentialDecayDelay', 2, 200.0, True,
                                           group='radiation_resistant')
    assert lab_s == lab_r and l_s == l_r
    assert u_s[:3] == [0.1, 0.1, 8.0]
    assert u_r[:3] == [0.1, 1.5, 8.0]
