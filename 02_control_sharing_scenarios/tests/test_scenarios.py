"""Tests of the partition of theta by scenario."""
import numpy as np
import pytest

from scenarios import (
    MODEL_PARAM_NAMES, SCENARIOS,
    count_model_params, ic_slice_for_group, scenario_id,
    split_model_args_for_group,
)

G = ['control_sensitive', 'control_resistant']
PN = MODEL_PARAM_NAMES['_log']


def test_scenario_id():
    assert scenario_id(2) == '_s2'


def test_count_model_params_exp():
    assert count_model_params(SCENARIOS['_exp'][1], G) == 2   # r per group
    assert count_model_params(SCENARIOS['_exp'][4], G) == 1   # r global


def test_count_model_params_log():
    assert count_model_params(SCENARIOS['_log'][1], G) == 4   # r and cc per group
    assert count_model_params(SCENARIOS['_log'][2], G) == 3   # cc global
    assert count_model_params(SCENARIOS['_log'][3], G) == 3   # r global
    assert count_model_params(SCENARIOS['_log'][4], G) == 2   # all global


def test_split_log_s1_all_group_specific():
    # theta = [r_sens, r_res, cc_sens, cc_res, T0s..., std]
    theta = np.array([0.10, 0.25, 1000., 2000., 80., 90., 100., 120., 50.])
    a_s, off = split_model_args_for_group(theta, SCENARIOS['_log'][1], PN, G,
                                          'control_sensitive')
    a_r, _ = split_model_args_for_group(theta, SCENARIOS['_log'][1], PN, G,
                                        'control_resistant')
    assert a_s == (0.10, 1000.0)
    assert a_r == (0.25, 2000.0)
    assert off == 4


def test_split_log_s2_cc_shared():
    # theta = [r_sens, r_res, cc_global, T0s..., std]
    theta = np.array([0.10, 0.25, 1800., 80., 90., 100., 120., 50.])
    a_s, _ = split_model_args_for_group(theta, SCENARIOS['_log'][2], PN, G,
                                        'control_sensitive')
    a_r, _ = split_model_args_for_group(theta, SCENARIOS['_log'][2], PN, G,
                                        'control_resistant')
    assert a_s == (0.10, 1800.0)      # same cc, different r
    assert a_r == (0.25, 1800.0)


def test_split_log_s3_r_shared():
    theta = np.array([0.15, 800., 2500., 80., 90., 100., 120., 50.])
    a_s, _ = split_model_args_for_group(theta, SCENARIOS['_log'][3], PN, G,
                                        'control_sensitive')
    a_r, _ = split_model_args_for_group(theta, SCENARIOS['_log'][3], PN, G,
                                        'control_resistant')
    assert a_s == (0.15, 800.0)       # same r, different cc
    assert a_r == (0.15, 2500.0)


def test_split_log_s4_all_shared():
    theta = np.array([0.15, 1500., 80., 90., 100., 120., 40.])
    a_s, _ = split_model_args_for_group(theta, SCENARIOS['_log'][4], PN, G,
                                        'control_sensitive')
    a_r, _ = split_model_args_for_group(theta, SCENARIOS['_log'][4], PN, G,
                                        'control_resistant')
    assert a_s == a_r == (0.15, 1500.0)


def test_ic_slice_extracts_the_right_block(synthetic_two_groups):
    fd = synthetic_two_groups
    theta = np.array([0.10, 0.25, 1000., 2000., 80., 90., 100., 120., 50.])
    np.testing.assert_array_equal(
        ic_slice_for_group(theta, 4, fd, G, 'control_sensitive'), [80., 90.])
    np.testing.assert_array_equal(
        ic_slice_for_group(theta, 4, fd, G, 'control_resistant'), [100., 120.])


def test_ic_slice_unknown_group_raises(synthetic_two_groups):
    with pytest.raises(KeyError):
        ic_slice_for_group(np.zeros(9), 4, synthetic_two_groups, G, 'unknown_group')
