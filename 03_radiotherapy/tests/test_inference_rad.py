"""Tests of the radiotherapy likelihood."""
import numpy as np
import pytest

from inference_rad import (
    log_likelihood, log_likelihood_pointwise, log_prior, log_probability,
)

G = 'radiation_sensitive'
MODEL = '_ExponentialDecayDelay'   # 3 rad params: alpha, beta, delay


def _theta_global(alpha, beta, delay, ics, std):
    return np.array([alpha, beta, delay, *ics, std])


def test_log_lik_finite(synthetic_rad_data, growth_fixed):
    theta = _theta_global(0.02, 0.05, 1.0, [100.0, 90.0], 30.0)
    ll = log_likelihood(theta, MODEL, synthetic_rad_data, G, growth_fixed)
    assert np.isfinite(ll)


def test_log_lik_bad_variance_is_neg_inf(synthetic_rad_data, growth_fixed):
    theta = _theta_global(0.02, 0.05, 1.0, [100.0, 90.0], 0.0)  # std=0
    assert log_likelihood(theta, MODEL, synthetic_rad_data, G,
                          growth_fixed) == -np.inf


def test_log_lik_handles_ode_exception(monkeypatch, synthetic_rad_data, growth_fixed):
    import inference_rad

    def boom(*a, **k):
        raise RuntimeError('falha')

    monkeypatch.setattr(inference_rad, 'solve_model_dispatch', boom)
    theta = _theta_global(0.02, 0.05, 1.0, [100.0, 90.0], 30.0)
    assert log_likelihood(theta, MODEL, synthetic_rad_data, G,
                          growth_fixed) == -np.inf


def test_mouse_index_uses_single_mouse(synthetic_rad_data, growth_fixed):
    theta = np.array([0.02, 0.05, 1.0, 100.0, 30.0])   # [rad..., ic, std]
    ll = log_likelihood(theta, MODEL, synthetic_rad_data, G, growth_fixed,
                        mouse_index=0)
    assert np.isfinite(ll)


def test_joint_equals_sum_of_singles(synthetic_rad_data, growth_fixed):
    """Consistency of the two modes: joint likelihood == sum of individual likelihoods."""
    alpha, beta, delay, std = 0.02, 0.05, 1.0, 30.0
    ic0, ic1 = 100.0, 90.0
    joint = log_likelihood(_theta_global(alpha, beta, delay, [ic0, ic1], std),
                           MODEL, synthetic_rad_data, G, growth_fixed)
    s = sum(log_likelihood(np.array([alpha, beta, delay, ic, std]), MODEL,
                           synthetic_rad_data, G, growth_fixed, mouse_index=i)
            for i, ic in enumerate([ic0, ic1]))
    np.testing.assert_allclose(joint, s, rtol=1e-12)


def test_pointwise_sums_to_scalar(synthetic_rad_data, growth_fixed):
    theta = _theta_global(0.02, 0.05, 1.0, [100.0, 90.0], 30.0)
    pw = log_likelihood_pointwise(theta, MODEL, synthetic_rad_data, G, growth_fixed)
    total = log_likelihood(theta, MODEL, synthetic_rad_data, G, growth_fixed)
    assert pw.shape == (18,)                    # 2 mice x 9 points
    assert pw.sum() == pytest.approx(total, rel=1e-12)


def test_uses_group_specific_growth(synthetic_rad_data, growth_fixed):
    """The likelihood uses r of the right group (sensitive versus resistant)."""
    theta = _theta_global(0.02, 0.05, 1.0, [110.0, 105.0], 30.0)
    ll_s = log_likelihood(theta, MODEL, synthetic_rad_data, 'radiation_sensitive',
                          growth_fixed)
    ll_r = log_likelihood(theta, MODEL, synthetic_rad_data, 'radiation_resistant',
                          growth_fixed)
    # different groups (different r and data) -> different ll
    assert ll_s != ll_r


def test_log_prior_box():
    l, u = [0.0, 0.0, 0.1], [0.1, 2.0, 10.0]
    assert log_prior([0.05, 1.0, 5.0], l, u) == 0.0
    assert log_prior([0.2, 1.0, 5.0], l, u) == -np.inf


def test_log_probability_neg_inf_outside(synthetic_rad_data, growth_fixed):
    theta = _theta_global(0.5, 0.05, 1.0, [100.0, 90.0], 30.0)   # alpha=0.5 fora
    l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    u = [0.1, 0.1, 8.0, 500., 500., 1000.]
    assert log_probability(theta, l, u, MODEL, synthetic_rad_data, G,
                           growth_fixed) == -np.inf
