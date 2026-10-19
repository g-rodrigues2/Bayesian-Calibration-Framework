"""Numerical parity with the original implementation.

The two original likelihood functions (global and mouse-specific) were replaced by a single
log_likelihood parametrized by ``mouse_index``. These tests reimplement the original formulas verbatim
and require exact equality with the current implementation.
"""
import numpy as np
import pytest

from inference import log_likelihood
from models import solve_model

GROUP = 'control_sensitive'


def _old_log_likelihood_global(theta, model_extension, full_data, group):
    """Original global-mode likelihood (verbatim)."""
    ll = 0
    variance = theta[-1] ** 2
    ndim = len(full_data[group])
    for idx, mouse_data in enumerate(full_data[group]):
        time_array = mouse_data['data'][:, 0]
        parameters = tuple(theta[:-(ndim + 1)])
        initial_condition = theta[-(ndim + 1) + idx]
        solution = solve_model(model_extension, time_array, parameters,
                               initial_condition)
        observed_volume = mouse_data['data'][:, 1]
        ll += -0.5 * np.sum(
            (solution[:, 0] - observed_volume) ** 2 / variance
            + np.log(2 * np.pi) + np.log(variance))
    return ll


def _old_log_likelihood_mouse(theta, model_extension, full_data, group,
                              mouse_index):
    """Original mouse-specific likelihood (verbatim)."""
    ll = 0
    variance = theta[-1] ** 2
    mouse_data = full_data[group][mouse_index]
    time_array = mouse_data['data'][:, 0]
    parameters = tuple(theta[:-2])
    initial_condition = theta[-2]
    solution = solve_model(model_extension, time_array, parameters,
                           initial_condition)
    observed_volume = mouse_data['data'][:, 1]
    ll += -0.5 * np.sum(
        (solution[:, 0] - observed_volume) ** 2 / variance
        + np.log(2 * np.pi) + np.log(variance))
    return ll


# theta = [parameters..., ic1, ic2, std] for the 2-mouse group of the fixture
GLOBAL_THETAS = [
    ('_exp',    np.array([0.10, 100.0, 150.0, 25.0])),
    ('_mendel', np.array([0.20, 0.90, 100.0, 150.0, 25.0])),
    ('_bert',   np.array([0.50, 0.05, 100.0, 150.0, 25.0])),
    ('_lin',    np.array([0.50, 500.0, 100.0, 150.0, 25.0])),
    ('_gomp',   np.array([0.10, 3000.0, 100.0, 100.0, 150.0, 25.0])),
    ('_surf',   np.array([0.50, 500.0, 100.0, 150.0, 25.0])),
    ('_log',    np.array([0.20, 2000.0, 100.0, 150.0, 25.0])),
]

# theta = [params..., ic, std]
MOUSE_THETAS = [
    ('_exp',    np.array([0.10, 100.0, 25.0])),
    ('_mendel', np.array([0.20, 0.90, 100.0, 25.0])),
    ('_bert',   np.array([0.50, 0.05, 100.0, 25.0])),
    ('_lin',    np.array([0.50, 500.0, 100.0, 25.0])),
    ('_gomp',   np.array([0.10, 3000.0, 100.0, 100.0, 25.0])),
    ('_surf',   np.array([0.50, 500.0, 100.0, 25.0])),
    ('_log',    np.array([0.20, 2000.0, 100.0, 25.0])),
]


@pytest.mark.parametrize('model_extension,theta', GLOBAL_THETAS)
def test_parity_global_mode(model_extension, theta, synthetic_full_data):
    """log_likelihood(mouse_index=None) reproduces the original formula."""
    current = log_likelihood(theta, model_extension, synthetic_full_data, GROUP)
    original = _old_log_likelihood_global(theta, model_extension,
                                       synthetic_full_data, GROUP)
    assert np.isfinite(original), 'the reference theta must give a finite ll'
    np.testing.assert_allclose(current, original, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize('model_extension,theta', MOUSE_THETAS)
@pytest.mark.parametrize('mouse_index', [0, 1])
def test_parity_mouse_specific_mode(model_extension, theta, mouse_index,
                                    synthetic_full_data):
    """log_likelihood(mouse_index=i) reproduces the original formula."""
    current = log_likelihood(theta, model_extension, synthetic_full_data, GROUP,
                          mouse_index=mouse_index)
    original = _old_log_likelihood_mouse(theta, model_extension,
                                      synthetic_full_data, GROUP, mouse_index)
    assert np.isfinite(original), 'the reference theta must give a finite ll'
    np.testing.assert_allclose(current, original, rtol=1e-12, atol=0.0)
