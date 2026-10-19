"""Tests of the radiotherapy sharing scenarios V1-V8."""
import numpy as np
import pytest

from scenarios_rad import (
    MODEL_PARAM_NAMES, SCENARIOS, WINNER_MODEL, count_model_params, describe, scenario_id,
    split_model_args_for_group,
)
from bounds_scen_rad import define_scenario_bounds_labels
from inference_scen_rad import (
    log_likelihood_shared, log_likelihood_shared_pointwise, log_probability_shared,
)

M = WINNER_MODEL
G = ['radiation_sensitive', 'radiation_resistant']
PN = MODEL_PARAM_NAMES[M]


def test_eight_scenarios():
    assert set(SCENARIOS[M]) == set(range(1, 9))


def test_scenario_id():
    assert scenario_id(4) == '_V4'


# number of model parameters per scenario (a, b, delay -> global=1, group=2)
@pytest.mark.parametrize('snum,expected', [
    (1, 6),  # all group-specific
    (2, 4),  # a global, b group, delay global -> 1+2+1
    (3, 4),  # a group, b global, delay global -> 2+1+1
    (4, 3),  # all global
    (5, 5),  # a, b group; delay global -> 2+2+1
    (6, 5),  # a group, b global, delay group -> 2+1+2
    (7, 5),  # a global, b group, delay group -> 1+2+2
    (8, 4),  # a, b global; delay group -> 1+1+2
])
def test_count_model_params(snum, expected):
    assert count_model_params(SCENARIOS[M][snum], G) == expected


def test_split_v1_all_group_specific():
    # theta = [a_s, a_r, b_s, b_r, d_s, d_r, ics..., std]
    theta = np.array([0.01, 0.02, 0.03, 0.04, 1.0, 2.0, 100., 90., 110., 105., 30.])
    a_s, _ = split_model_args_for_group(theta, SCENARIOS[M][1], PN, G, 'radiation_sensitive')
    a_r, _ = split_model_args_for_group(theta, SCENARIOS[M][1], PN, G, 'radiation_resistant')
    assert a_s == (0.01, 0.03, 1.0)
    assert a_r == (0.02, 0.04, 2.0)


def test_split_v4_all_global():
    theta = np.array([0.01, 0.03, 1.5, 100., 90., 110., 105., 30.])
    a_s, _ = split_model_args_for_group(theta, SCENARIOS[M][4], PN, G, 'radiation_sensitive')
    a_r, _ = split_model_args_for_group(theta, SCENARIOS[M][4], PN, G, 'radiation_resistant')
    assert a_s == a_r == (0.01, 0.03, 1.5)


def test_split_v8_delay_group_specific():
    # a, b global; delay per group. theta=[a, b, d_s, d_r, ics..., std]
    theta = np.array([0.01, 0.03, 1.0, 2.0, 100., 90., 110., 105., 30.])
    a_s, _ = split_model_args_for_group(theta, SCENARIOS[M][8], PN, G, 'radiation_sensitive')
    a_r, _ = split_model_args_for_group(theta, SCENARIOS[M][8], PN, G, 'radiation_resistant')
    assert a_s == (0.01, 0.03, 1.0)
    assert a_r == (0.01, 0.03, 2.0)   # same a and b; different delay


def test_describe():
    d = describe(M, 4)
    assert 'shared' in d
    d1 = describe(M, 1)
    assert 'group-specific' in d1


# === bounds ===

@pytest.mark.parametrize('snum', range(1, 9))
def test_bounds_ndim(snum, rad_two_groups):
    l, u, lab = define_scenario_bounds_labels(M, SCENARIOS[M][snum], G,
                                              rad_two_groups, max_ic=200.0)
    n_model = count_model_params(SCENARIOS[M][snum], G)
    n_mice = 4
    assert len(l) == len(u) == len(lab) == n_model + n_mice + 1
    assert lab[-1] == 'std'
    assert all(lo < hi for lo, hi in zip(l, u))


def test_bounds_labels_v2(rad_two_groups):
    l, u, lab = define_scenario_bounds_labels(M, SCENARIOS[M][2], G,
                                              rad_two_groups, max_ic=200.0)
    # a global, b per group, delay global
    assert lab[:4] == ['alpha_rad_global', 'beta_rad_sensitive',
                       'beta_rad_resistant', 'delay_global']


# === inference ===

def test_shared_lik_finite(rad_two_groups, growth_fixed):
    l, u, _ = define_scenario_bounds_labels(M, SCENARIOS[M][4], G, rad_two_groups, 200.)
    theta = np.array([0.02, 0.05, 1.0, 100., 90., 110., 105., 30.])
    ll = log_likelihood_shared(theta, M, rad_two_groups, G, SCENARIOS[M][4], growth_fixed)
    assert np.isfinite(ll)


def test_shared_pointwise_sums_to_scalar(rad_two_groups, growth_fixed):
    theta = np.array([0.02, 0.05, 1.0, 100., 90., 110., 105., 30.])
    pw = log_likelihood_shared_pointwise(theta, M, rad_two_groups, G,
                                         SCENARIOS[M][4], growth_fixed)
    total = log_likelihood_shared(theta, M, rad_two_groups, G,
                                    SCENARIOS[M][4], growth_fixed)
    assert pw.shape == (24,)   # 4 mice x 6 points
    assert pw.sum() == pytest.approx(total, rel=1e-12)


def test_shared_lik_bad_variance(rad_two_groups, growth_fixed):
    theta = np.array([0.02, 0.05, 1.0, 100., 90., 110., 105., 0.0])  # std=0
    assert log_likelihood_shared(theta, M, rad_two_groups, G, SCENARIOS[M][4],
                                 growth_fixed) == -np.inf


def test_log_probability_outside_box(rad_two_groups, growth_fixed):
    sdef = SCENARIOS[M][4]
    theta = np.array([0.5, 0.05, 1.0, 100., 90., 110., 105., 30.])  # alpha outside the bounds
    l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    u = [0.1, 0.1, 8.0, 500., 500., 500., 500., 1000.]
    assert log_probability_shared(theta, l, u, M, rad_two_groups, G, sdef,
                                  growth_fixed) == -np.inf


def test_scenario_bounds_follow_thesis_priors(rad_two_groups):
    """alpha ~ U(0, 1), beta ~ U(0, 1.5), delay ~ U(0, 8) (Table 4.1 of the thesis)."""
    l, u, lab = define_scenario_bounds_labels(M, SCENARIOS[M][2], G, rad_two_groups, 200.)
    bounds = dict(zip(lab, zip(l, u)))
    assert bounds['alpha_rad_global'] == (0.0, 1.0)
    assert bounds['beta_rad_sensitive'] == bounds['beta_rad_resistant'] == (0.0, 1.5)
    assert bounds['delay_global'] == (0.0, 8.0)
