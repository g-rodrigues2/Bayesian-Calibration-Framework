import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bounds_ri import define_bounds_labels  # noqa: E402
from data_loader_ri import load_data  # noqa: E402
from inference_ri import log_likelihood, split_theta  # noqa: E402
from rad_immuno_models import (covariate_value, lysis_coefficient, solve_mouse,  # noqa: E402
                                standardized_upstream)
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


def test_standardized_models_are_positive_for_any_coefficients():
    """exp(a z + b) cannot produce the negative lysis that the linear form allows."""
    mice = load_data()['rad+io_sensitive']
    for model in ('_SUVExp', '_VolExp'):
        up = standardized_upstream(model, UP, mice)
        for params in ((-3.0, -4.61), (3.0, 0.69), (0.0, -4.61)):
            phis = [lysis_coefficient(model, params, covariate_value(model, m), up) for m in mice]
            assert all(p > 0 for p in phis)
    linear = lysis_coefficient('_SUVLinear', (-1.0, 0.5), 1.2, UP)
    assert linear < 0


def test_standardization_uses_the_training_fold_only():
    mice = load_data()['rad+io_sensitive']
    held_out = mice[0]
    training = [m for m in mice if m['name'] != held_out['name']]
    up_all = standardized_upstream('_SUVExp', UP, mice)
    up_fold = standardized_upstream('_SUVExp', UP, training)
    assert up_fold['cov_mean'] != up_all['cov_mean']
    values = np.array([covariate_value('_SUVExp', m) for m in training])
    assert np.isclose(up_fold['cov_mean'], values.mean())
    assert np.isclose(up_fold['cov_std'], values.std(ddof=1))


def test_volume_covariate_is_the_first_observation():
    mice = load_data()['rad+io_sensitive']
    assert all(covariate_value('_VolExp', m) == m['data'][0, 1] for m in mice)
    assert covariate_value('_SUVExp', mice[0]) == mice[0]['SUV']


def test_constant_covariate_is_rejected():
    flat = [dict(_mouse(), name=f'm{i}', SUV=1.2) for i in range(3)]
    try:
        standardized_upstream('_SUVExp', UP, flat)
    except ValueError as exc:
        assert 'constant' in str(exc)
    else:
        raise AssertionError('a constant covariate should be rejected')


def test_standardized_model_matches_the_compiled_solver():
    mice = load_data()['rad+io_sensitive'][:4]
    up = standardized_upstream('_SUVExp', UP, mice)
    theta = [0.4, -1.2] + [float(m['data'][0, 1]) for m in mice] + [80.0]
    compiled = log_likelihood(theta, '_SUVExp', mice, up)
    params, ics, std = split_theta(theta, '_SUVExp', len(mice))
    reference = 0.0
    for mouse, ic in zip(mice, ics):
        pred = solve_mouse('_SUVExp', params, ic, mouse, up)
        reference += float(-0.5 * np.sum((pred - mouse['data'][:, 1]) ** 2 / std ** 2
                                         + np.log(2 * np.pi * std ** 2)))
    assert abs(compiled - reference) < 1e-3


def test_profiled_fit_is_self_consistent_and_beats_a_plain_local_search():
    """fast_fit profiles out the std and the initial volumes; the theta it returns must reproduce
    the log-likelihood it reports, and must clear the narrow optimum that a single bounded search
    over the whole range misses."""
    import fast_fit
    from scipy.optimize import minimize_scalar

    mice = load_data()['rad+io_sensitive']
    training = [m for m in mice if m['name'] != 'c0_m10']
    ic_hi = fast_fit.IC_UPPER_FACTOR * max(float(m['data'][0, 1]) for m in mice)
    theta, ll = fast_fit.fit('_RadImmunoFixed', training, UP, ic_hi)

    assert len(theta) == len(training) + 1
    assert abs(log_likelihood(theta, '_RadImmunoFixed', training, UP) - ll) < 1e-3

    import rad_immuno_fast as fast
    cohort = fast.cohort(training)
    grid = cohort.grid(UP['delay'])
    m = [i for i, x in enumerate(training) if x['name'] == 'c0_m26'][0]

    def sse(ic):
        return fast_fit._sse_of_mouse(cohort, m, ic, UP['d'], UP, grid)

    naive = minimize_scalar(sse, bounds=(1e-6, ic_hi), method='bounded',
                            options={'xatol': 1e-8, 'maxiter': 200})
    assert sse(theta[m]) < naive.fun          # the scan clears the well the local search steps over


def test_inherited_immunotherapy_vector_is_the_calibrated_sample():
    """The inherited d, s and lam are one sample of the immunotherapy chain, the one of largest
    likelihood, and not a mix of summaries of different runs. Re-deriving them from the stored
    calibration keeps thesis_params.py tied to the stage it claims to inherit from."""
    import thesis_params
    tnbc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stage = os.path.join(tnbc, '05_immunotherapy', 'Output_Calibration')
    stored = np.load(os.path.join(stage, 'multi_ll_pars_io_sensitive_dePillisRadunskayaLaw.npz'))
    sample = dict(zip([str(x) for x in stored['labels']], stored['pars']))
    for name in ('d', 's', 'lam'):
        assert thesis_params.IMMUNOTHERAPY['sensitive'][name] == float(sample[name])
