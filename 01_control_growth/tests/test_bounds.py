"""Tests of the prior bounds (uniform priors) and labels."""
import pytest

from bounds import define_bounds_labels, find_max_initial_condition


# (model, number of growth parameters)
MODEL_NPARS = [
    ('_exp', 1), ('_mendel', 2), ('_bert', 2), ('_lin', 2),
    ('_gomp', 3), ('_surf', 2), ('_log', 2),
]


@pytest.mark.parametrize('model,npars', MODEL_NPARS)
def test_ndim_global_mode(model, npars):
    """Modo global: ndim = n_params + group_size (ICs) + 1 (std)."""
    l, u, labels = define_bounds_labels(model, group_size=5, max_ic=150.0,
                                        bool_multiple_mice=True)
    assert len(l) == len(u) == len(labels) == npars + 5 + 1
    assert labels[-1] == 'std'
    assert labels[npars] == 'ic1'


@pytest.mark.parametrize('model,npars', MODEL_NPARS)
def test_ndim_mouse_specific_mode(model, npars):
    """Modo mouse-specific: ndim = n_params + 1 (ic) + 1 (std)."""
    l, u, labels = define_bounds_labels(model, group_size=5, max_ic=150.0,
                                        bool_multiple_mice=False)
    assert len(l) == len(u) == len(labels) == npars + 2
    assert labels[-2:] == ['ic', 'std']


@pytest.mark.parametrize('model,_npars', MODEL_NPARS)
def test_lower_strictly_below_upper(model, _npars):
    l, u, _ = define_bounds_labels(model, group_size=3, max_ic=150.0,
                                   bool_multiple_mice=True)
    assert all(lo < hi for lo, hi in zip(l, u))


def test_unknown_model_raises():
    with pytest.raises(ValueError, match='Unknown model'):
        define_bounds_labels('_unknown', group_size=1, max_ic=100.0)


def test_ic_and_std_bounds():
    """Initial conditions up to 2*max_ic; std in [0.1, 1000]."""
    max_ic = 147.05405
    l, u, labels = define_bounds_labels('_log', group_size=2, max_ic=max_ic,
                                        bool_multiple_mice=True)
    ic_idx = [i for i, name in enumerate(labels) if name.startswith('ic')]
    for i in ic_idx:
        assert l[i] == 0.0
        assert u[i] == pytest.approx(2 * max_ic)
    assert (l[-1], u[-1]) == (0.1, 1000.0)


def test_log_bounds_unified_on_global_table():
    """_log uses the same bounds in both modes: r in [0, 1], cc in [100, 20000]."""
    for multiple in (True, False):
        l, u, labels = define_bounds_labels('_log', group_size=2, max_ic=100.0,
                                            bool_multiple_mice=multiple)
        assert labels[:2] == ['r', 'cc']
        assert (l[0], u[0]) == (0.0, 1.0)
        assert (l[1], u[1]) == (100.0, 20000)


def test_gomp_bounds_scale_with_ic_max():
    """_gomp: b up to 10*ic_max and c up to ic_max."""
    max_ic = 200.0
    ic_max = 2 * max_ic
    l, u, _ = define_bounds_labels('_gomp', group_size=1, max_ic=max_ic)
    assert u[1] == pytest.approx(10 * ic_max)
    assert u[2] == pytest.approx(ic_max)


def test_find_max_initial_condition(synthetic_full_data):
    """Returns the largest T(0) and where it comes from."""
    max_ic, group, mouse = find_max_initial_condition(synthetic_full_data)
    # control_resistant c1_m1 has T(0) = 100 * 2.0 = 200
    assert max_ic == pytest.approx(200.0)
    assert group == 'control_resistant'
    assert mouse == 'c1_m1'
