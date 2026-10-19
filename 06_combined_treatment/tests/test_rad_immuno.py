import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bounds_ri import define_bounds_labels  # noqa: E402
from data_loader_ri import load_data  # noqa: E402
from inference_ri import log_likelihood, split_theta  # noqa: E402
from rad_immuno_models import lysis_coefficient, solve_mouse  # noqa: E402
from thesis_params import upstream_params  # noqa: E402

UP = upstream_params('sensitive')


def _mouse(t_end=30.0):
    t = np.arange(0.0, t_end + 1, 1.0)
    return dict(name='m', data=np.column_stack([t, np.full_like(t, 100.0)]), SUV=1.2,
                rad_days=np.array([0.0, 1, 2, 3, 4, 5]), io_days=np.array([9.0, 12, 15, 18, 21, 24]))


def test_data_cohort():
    mice = load_data()['rad+io_sensitive']
    assert len(mice) == 15 and all(m['SUV'] is not None for m in mice)


def test_logistic_solution_without_treatment():
    up = dict(UP, a_rad=0.0, d=0.0)
    mouse = _mouse()
    T = solve_mouse('_RadImmunoFixed', (), 100.0, mouse, up)
    t = mouse['data'][:, 0]
    exact = up['cc'] / (1 + (up['cc'] / 100.0 - 1) * np.exp(-up['r'] * t))
    assert np.allclose(T, exact, rtol=1e-5)


def test_linear_suv_equals_fixed_when_phi_matches_d():
    mouse = _mouse()
    a = UP['d'] / mouse['SUV']
    assert np.isclose(lysis_coefficient('_SUVLinear', (a, 0.0), mouse['SUV'], UP), UP['d'])
    assert np.allclose(solve_mouse('_SUVLinear', (a, 0.0), 150.0, mouse, UP),
                       solve_mouse('_RadImmunoFixed', (), 150.0, mouse, UP))


def test_compiled_solver_matches_odeint():
    import rad_immuno_fast
    mice = load_data()['rad+io_sensitive'][:4]
    rng = np.random.default_rng(0)
    for model, params in (('_RadImmunoFixed', ()), ('_SUVLinear', (1.1, 0.2)), ('_SUVLinear', (-0.3, 0.1))):
        ics = rng.uniform(50.0, 200.0, len(mice))
        for mouse, ic in zip(mice, ics):
            fast = rad_immuno_fast.predict(model, params, ic, mouse, UP)
            slow = solve_mouse(model, params, ic, mouse, UP)
            assert np.allclose(fast, slow, rtol=1e-5, atol=1e-3)
        theta = np.concatenate([params, ics, [80.0]])
        ll_odeint = sum(-0.5 * np.sum((solve_mouse(model, params, ic, m, UP) - m['data'][:, 1]) ** 2 / 80.0 ** 2
                                      + np.log(2 * np.pi * 80.0 ** 2)) for m, ic in zip(mice, ics))
        assert abs(log_likelihood(theta, model, mice, UP) - ll_odeint) < 1e-3


def test_theta_layout_and_likelihood():
    mice = [_mouse(), _mouse()]
    l_bound, u_bound, labels = define_bounds_labels('_SUVLinear', 2, 100.0, 'suv')
    assert labels == ['a', 'b', 'ic1', 'ic2', 'std'] and len(l_bound) == len(u_bound) == 5
    theta = np.array([1.0, 0.0, 100.0, 100.0, 50.0])
    params, ics, std = split_theta(theta, '_SUVLinear', 2)
    assert params == (1.0, 0.0) and list(ics) == [100.0, 100.0] and std == 50.0
    assert np.isfinite(log_likelihood(theta, '_SUVLinear', mice, UP))
