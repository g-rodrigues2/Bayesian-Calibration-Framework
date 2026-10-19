"""Tests of the immunotherapy stage: models, bounds (mixed parameterization), data loading, inference
and prior predictive check."""
import os

import numpy as np
import pytest

import immuno_models as im
from bounds_imm import (
    CALIBRATABLE_MODELS, count_treatment_slots,
    define_bounds_labels, extract_treatment_params, find_max_initial_condition,
)
from data_loader_imm import (
    _FNAME_RE, load_immuno_data, parse_dose_days_env,
)
from inference_imm import (
    log_likelihood, log_likelihood_pointwise, log_prior, log_probability,
)
from prior_predictive import prior_predictive_coverage, sample_prior

G = 'io_sensitive'
M = '_dePillisRadunskayaLaw'
GROWTH = (0.088, 3741.0)


# === models ===

def test_thesis_models_present():
    """The 12 formulations of Table S3.1 of the thesis."""
    assert len(im.ALL_MODELS) == 12
    for m in im.ALL_MODELS:
        assert hasattr(im, 'model' + m)
    assert im.WINNER_MODEL == M


def test_every_model_has_bounds():
    assert CALIBRATABLE_MODELS == im.ALL_MODELS


@pytest.mark.parametrize('model', CALIBRATABLE_MODELS)
def test_every_calibratable_model_integrates(model, dose_days):
    t = np.array([0., 5., 10., 20., 30.])
    n_p = im.n_treatment_params(model)
    l, u, _ = define_bounds_labels(model, 3, 200.0, True)
    mid = [(a + b) / 2 for a, b in zip(l[:n_p], u[:n_p])]
    sol = im.solve_model(model, t, (*GROWTH, *mid), 100.0, dose_days, 1.0)
    assert sol.shape == (5, 1)
    assert np.all(np.isfinite(sol))


def test_unknown_model_raises():
    with pytest.raises(ValueError, match='Unknown immunotherapy model'):
        define_bounds_labels('_PowerForm', 3, 200.0, True)


def test_treatment_reduces_volume(dose_days):
    """With an effect (d > 0) the final volume is smaller than without."""
    t = np.array([0., 15., 30.])
    with_io = im.solve_model(M, t, (*GROWTH, 0.8, 0.5, 1.0), 100., dose_days, 1.)[-1, 0]
    without_io = im.solve_model(M, t, (*GROWTH, 0.0, 0.5, 1.0), 100., dose_days, 1.)[-1, 0]
    assert with_io < without_io


def test_widened_cd_acd_priors():
    """beta_immuno upper bounds of CD and ACD widened after the defense."""
    _l, u, lab = define_bounds_labels('_CumulativeDose', 3, 200.0, True)
    assert u[lab.index('beta_immuno')] == 15.0
    _l, u, lab = define_bounds_labels('_AccelCD', 3, 200.0, True)
    assert u[lab.index('beta_immuno')] == 100.0


def test_application_days_matter(dose_days):
    t = np.array([0., 15., 30.])
    p = (*GROWTH, 0.8, 0.5, 1.0)
    early = im.solve_model(M, t, p, 100., np.array([0, 3, 6]), 1.)[-1, 0]
    late = im.solve_model(M, t, p, 100., np.array([20, 23, 26]), 1.)[-1, 0]
    assert abs(early - late) > 1.0


# === bounds and mixed parameterization ===

def test_bounds_global_no_ms():
    l, u, lab = define_bounds_labels(M, 3, 200.0, True, ms_params=())
    assert lab[:3] == ['d', 's', 'lam']
    assert len(l) == 3 + 3 + 1            # parameters + 3 initial conditions + std
    assert lab[-1] == 'std'


def test_bounds_hybrid_expands_only_ms_params():
    l, u, lab = define_bounds_labels(M, 3, 200.0, True, ms_params=('d',))
    assert lab[:5] == ['d1', 'd2', 'd3', 's', 'lam']
    assert len(l) == 5 + 3 + 1
    # d1..d3 have the bounds of d
    assert l[0] == l[1] == l[2]


def test_bounds_mouse_mode_ignores_ms():
    l, u, lab = define_bounds_labels(M, 3, 200.0, False, ms_params=('d',))
    assert lab == ['d', 's', 'lam', 'ic', 'std']


def test_count_treatment_slots():
    assert count_treatment_slots(M, 3, ()) == 3
    assert count_treatment_slots(M, 3, ('d',)) == 5          # 3 + 1 + 1
    assert count_treatment_slots(M, 3, ('d', 'lam')) == 7    # 3 + 1 + 3


def test_extract_treatment_params_hybrid():
    # theta = [d1, d2, d3, s, lam, ...]
    theta = np.array([0.1, 0.2, 0.3, 0.5, 1.0, 100., 110., 95., 30.])
    p0, n = extract_treatment_params(theta, M, 3, 0, ('d',))
    p2, _ = extract_treatment_params(theta, M, 3, 2, ('d',))
    assert p0 == (0.1, 0.5, 1.0)
    assert p2 == (0.3, 0.5, 1.0)     # only d changes
    assert n == 5


def test_lower_below_upper_all_models():
    for m in CALIBRATABLE_MODELS:
        l, u, _ = define_bounds_labels(m, 3, 200.0, True)
        assert all(a < b for a, b in zip(l, u))


def test_find_max_initial_condition(synthetic_imm_data):
    assert find_max_initial_condition(synthetic_imm_data) == pytest.approx(110.0)


# === data loading ===

_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
_HAS = os.path.isdir(_DATA) and bool(os.listdir(_DATA))


@pytest.mark.skipif(not _HAS, reason='data folder not found')
def test_loader_groups_and_suv():
    fd = load_immuno_data(_DATA, dose_days=[0, 3, 6, 9, 12, 15, 18])
    assert set(fd) == {'io_sensitive'}
    assert len(fd['io_sensitive']) == 3
    suvs = sorted(m['SUV'] for m in fd['io_sensitive'])
    assert suvs == pytest.approx([0.728827738, 1.16068487, 1.202419278])


@pytest.mark.skipif(not _HAS, reason='data folder not found')
def test_loader_default_is_real_schedule():
    # The default is the administration schedule of the experiment (not a single dose at day 0);
    # CALIB_DOSE_DAYS overrides it.
    fd = load_immuno_data(_DATA)
    for m in fd['io_sensitive']:
        np.testing.assert_array_equal(m['treatment_days'],
                                      [0, 3, 6, 9, 12, 15, 18])


@pytest.mark.skipif(not _HAS, reason='data folder not found')
def test_loader_dose_days_override():
    fd = load_immuno_data(_DATA, dose_days=[0, 3, 6])
    for m in fd['io_sensitive']:
        np.testing.assert_array_equal(m['treatment_days'], [0, 3, 6])


def test_regex_tolerates_suv_suffix():
    assert _FNAME_RE.match('io_sensitive_c0_m3_t0_t3_SUV0.72') is not None
    assert _FNAME_RE.match('io_resistant_c1_m9') is not None
    assert _FNAME_RE.match('control_sensitive_c0_m1') is None


def test_parse_dose_days_env():
    np.testing.assert_array_equal(parse_dose_days_env('0,3,6,9,12,15,18'),
                                  [0, 3, 6, 9, 12, 15, 18])
    assert parse_dose_days_env('') is None


# === inference ===

def _theta_group(d, s, lam, ics, std):
    return np.array([d, s, lam, *ics, std])


def test_log_lik_finite(synthetic_imm_data, growth_fixed):
    th = _theta_group(0.5, 0.5, 1.0, [100., 110., 95.], 25.)
    assert np.isfinite(log_likelihood(th, M, synthetic_imm_data, G, growth_fixed))


def test_log_lik_bad_std(synthetic_imm_data, growth_fixed):
    th = _theta_group(0.5, 0.5, 1.0, [100., 110., 95.], 0.)
    assert log_likelihood(th, M, synthetic_imm_data, G, growth_fixed) == -np.inf


def test_joint_equals_sum_of_singles(synthetic_imm_data, growth_fixed):
    """Without ms_params, the group likelihood equals the sum of the individual likelihoods."""
    d, s, lam, std = 0.5, 0.5, 1.0, 25.
    ics = [100., 110., 95.]
    joint = log_likelihood(_theta_group(d, s, lam, ics, std), M,
                           synthetic_imm_data, G, growth_fixed)
    total = sum(log_likelihood(np.array([d, s, lam, ic, std]), M,
                               synthetic_imm_data, G, growth_fixed, mouse_index=i)
                for i, ic in enumerate(ics))
    np.testing.assert_allclose(joint, total, rtol=1e-12)


def test_hybrid_uses_per_mouse_param(synthetic_imm_data, growth_fixed):
    """With ms_params=('d',), each mouse uses its own d."""
    theta = np.array([0.1, 0.9, 0.5, 0.5, 1.0, 100., 110., 95., 25.])
    ll = log_likelihood(theta, M, synthetic_imm_data, G, growth_fixed,
                        ms_params=('d',))
    assert np.isfinite(ll)
    # changing d1 changes the likelihood
    t2 = theta.copy(); t2[0] = 1.9
    assert log_likelihood(t2, M, synthetic_imm_data, G, growth_fixed,
                          ms_params=('d',)) != ll


def test_pointwise_sums_to_scalar(synthetic_imm_data, growth_fixed):
    th = _theta_group(0.5, 0.5, 1.0, [100., 110., 95.], 25.)
    pw = log_likelihood_pointwise(th, M, synthetic_imm_data, G, growth_fixed)
    total = log_likelihood(th, M, synthetic_imm_data, G, growth_fixed)
    assert pw.shape == (24,)                 # 3 mice x 8 points
    assert pw.sum() == pytest.approx(total, rel=1e-12)


def test_log_prior_and_probability(synthetic_imm_data, growth_fixed):
    l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u = [2.0, 2.0, 2.0, 500., 500., 500., 200.]
    th = _theta_group(0.5, 0.5, 1.0, [100., 110., 95.], 25.)
    assert log_prior(th, l, u) == 0.0
    outside = th.copy(); outside[0] = 9.0
    assert log_probability(outside, l, u, M, synthetic_imm_data, G,
                           growth_fixed) == -np.inf


# === prior predictive check ===

def test_sample_prior_inside_box():
    l, u, _ = define_bounds_labels(M, 3, 200.0, True)
    th = sample_prior(l, u, 50, np.random.default_rng(0))
    assert th.shape == (50, len(l))
    assert np.all(th >= np.array(l)) and np.all(th <= np.array(u))


def test_prior_predictive_coverage_returns_both(synthetic_imm_data, growth_fixed):
    l, u, _ = define_bounds_labels(M, 3, 200.0, True)
    th = sample_prior(l, u, 60, np.random.default_rng(1))
    cov = prior_predictive_coverage(M, th, growth_fixed, G,
                                    synthetic_imm_data[G][0], 0, 3)
    assert cov is not None
    fp, kp, fc, kc, n = cov
    assert 0.0 <= fc <= 1.0 and 0.0 <= fp <= 1.0
    assert n == 8
    # the predictive band (with noise) contains at least as many points as the curve band
    assert kp >= kc


def test_logscale_posterior_matches_with_jacobian(synthetic_imm_data, growth_fixed):
    """log p(phi) = log p(theta) + sum(phi[log_index]) and the transforms are inverse."""
    from inference_imm import from_sampling, log_probability_logscale, to_sampling
    l, u, _ = define_bounds_labels(M, 3, 200.0, True)
    theta = _theta_group(0.5, 0.2, 1.0, [100., 110., 95.], 25.)
    idx = [0, 1]
    phi = to_sampling(theta, idx)
    np.testing.assert_allclose(from_sampling(phi, idx), theta)
    lp = log_probability(theta, l, u, M, synthetic_imm_data, G, growth_fixed)
    lp_log = log_probability_logscale(phi, l, u, idx, M, synthetic_imm_data, G, growth_fixed)
    assert lp_log == pytest.approx(lp + np.log(0.5) + np.log(0.2))
