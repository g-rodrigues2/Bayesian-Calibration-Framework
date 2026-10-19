"""Tests of the multi-group likelihood (total and pointwise)."""
import numpy as np
import pytest

from inference_shared import (
    log_likelihood_shared, log_likelihood_shared_pointwise,
    log_prior_uniform, log_probability_shared,
)
from scenarios import SCENARIOS

G = ['control_sensitive', 'control_resistant']


def test_log_lik_finite_inside_bounds(synthetic_two_groups):
    theta = np.array([0.10, 100.0, 50.0, 80.0, 120.0, 50.0])  # _exp s4
    ll = log_likelihood_shared(theta, '_exp', synthetic_two_groups, G,
                               SCENARIOS['_exp'][4])
    assert np.isfinite(ll)


def test_log_lik_prefers_true_group_specific_rates(synthetic_two_groups):
    """The fixture has r=0.1 (sensitive) and r=0.2 (resistant); scenario 1 with the true rates must
    beat scenario 1 with a wrong common rate."""
    sdef = SCENARIOS['_exp'][1]
    true = np.array([0.10, 0.20, 100., 50., 80., 120., 1.0])
    wrong = np.array([0.15, 0.15, 100., 50., 80., 120., 1.0])
    ll_true = log_likelihood_shared(true, '_exp', synthetic_two_groups, G, sdef)
    ll_wrong = log_likelihood_shared(wrong, '_exp', synthetic_two_groups, G, sdef)
    assert ll_true > ll_wrong


def test_pointwise_sums_to_scalar(synthetic_two_groups):
    """Invariant: sum_i log p(y_i|theta) == log_likelihood_shared."""
    theta = np.array([0.10, 0.20, 100., 50., 80., 120., 5.0])
    sdef = SCENARIOS['_exp'][1]
    pw = log_likelihood_shared_pointwise(theta, '_exp', synthetic_two_groups,
                                         G, sdef)
    total = log_likelihood_shared(theta, '_exp', synthetic_two_groups, G, sdef)
    assert pw.shape == (24,)          # 4 mice x 6 points
    assert pw.sum() == pytest.approx(total, rel=1e-12)


def test_pointwise_length_log_s4(synthetic_two_groups):
    theta = np.array([0.15, 1500., 100., 50., 80., 120., 5.0])
    pw = log_likelihood_shared_pointwise(theta, '_log', synthetic_two_groups,
                                         G, SCENARIOS['_log'][4])
    assert pw.shape == (24,)
    assert pw.sum() == pytest.approx(
        log_likelihood_shared(theta, '_log', synthetic_two_groups, G,
                              SCENARIOS['_log'][4]), rel=1e-12)


def test_pointwise_none_on_bad_variance(synthetic_two_groups):
    theta = np.array([0.10, 100.0, 50.0, 80.0, 120.0, 0.0])  # std=0
    assert log_likelihood_shared_pointwise(
        theta, '_exp', synthetic_two_groups, G, SCENARIOS['_exp'][4]) is None


def test_log_prior_box():
    l, u = [0.0, 0.0, 0.1], [1.0, 100.0, 10.0]
    assert log_prior_uniform([0.5, 50.0, 1.0], l, u) == 0.0
    assert log_prior_uniform([0.5, 150.0, 1.0], l, u) == -np.inf


def test_log_probability_neg_inf_outside_box(synthetic_two_groups):
    sdef = SCENARIOS['_exp'][4]
    theta = np.array([0.5, 100., 100., 100., 100., 0.0])   # std=0 outside the bound
    l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    u = [1.0, 500., 500., 500., 500., 100.]
    assert log_probability_shared(theta, l, u, '_exp', synthetic_two_groups,
                                  G, sdef) == -np.inf
