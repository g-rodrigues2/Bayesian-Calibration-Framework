"""Tests of the walker initialization (ball around the MAP estimate versus uniform)."""
import numpy as np
import pytest

from initialization import (
    ball_positions,
    find_map_estimate,
    initial_positions,
    uniform_positions,
)

# Synthetic target: narrow Gaussian around theta* inside a wide box
THETA_STAR = np.array([0.10, 4000.0, 110.0])
L = [0.0, 100.0, 0.1]
U = [1.0, 20000.0, 1000.0]


def _log_prob(theta, scale=np.array([0.01, 200.0, 10.0])):
    theta = np.asarray(theta, float)
    if np.any(theta <= np.asarray(L)) or np.any(theta >= np.asarray(U)):
        return -np.inf
    return -0.5 * np.sum(((theta - THETA_STAR) / scale) ** 2)


def _dentro(pos):
    return np.all(pos > np.asarray(L)) and np.all(pos < np.asarray(U))


# === uniform_positions ===

def test_uniform_positions_shape_and_bounds():
    rng = np.random.default_rng(0)
    pos = uniform_positions(64, L, U, rng)
    assert pos.shape == (64, 3)
    assert _dentro(pos)


def test_uniform_positions_spreads_over_the_whole_box():
    """Uniform initialization: half of the walkers start with std > 500."""
    rng = np.random.default_rng(1)
    pos = uniform_positions(2000, L, U, rng)
    assert (pos[:, 2] > 500).mean() == pytest.approx(0.5, abs=0.05)


# === find_map_estimate ===

def test_find_map_estimate_recovers_the_optimum():
    theta = find_map_estimate(_log_prob, (), L, U, n_random=300,
                              rng=np.random.default_rng(2))
    assert theta is not None
    np.testing.assert_allclose(theta, THETA_STAR, rtol=0.05)


def test_find_map_estimate_stays_inside_the_box():
    theta = find_map_estimate(_log_prob, (), L, U, n_random=100,
                              rng=np.random.default_rng(3))
    assert np.all(theta > np.asarray(L)) and np.all(theta < np.asarray(U))


def test_find_map_estimate_none_when_nothing_is_finite():
    def sempre_inf(_t):
        return -np.inf
    assert find_map_estimate(sempre_inf, (), L, U, n_random=20,
                             rng=np.random.default_rng(4)) is None


def test_find_map_estimate_never_worse_than_best_random_draw():
    rng = np.random.default_rng(5)
    theta = find_map_estimate(_log_prob, (), L, U, n_random=50, rng=rng)
    sorteios = np.random.default_rng(5).uniform(L, U, size=(50, 3))
    melhor_sorteio = max(_log_prob(t) for t in sorteios)
    assert _log_prob(theta) >= melhor_sorteio


# === ball_positions ===

def test_ball_positions_are_tight_around_theta0():
    pos = ball_positions(THETA_STAR, 200, L, U, _log_prob, (), scale=1e-3,
                         rng=np.random.default_rng(6))
    assert pos.shape == (200, 3)
    assert _dentro(pos)
    # relative deviation ~ scale in each coordinate
    rel = np.std(pos, axis=0) / np.abs(THETA_STAR)
    np.testing.assert_allclose(rel, 1e-3, rtol=0.5)


def test_ball_positions_all_have_finite_log_prob():
    pos = ball_positions(THETA_STAR, 100, L, U, _log_prob, (),
                         rng=np.random.default_rng(7))
    assert all(np.isfinite(_log_prob(p)) for p in pos)


def test_ball_positions_clip_when_theta0_hugs_a_bound():
    theta0 = np.array([0.10, 4000.0, 0.11])      # std at the lower bound
    pos = ball_positions(theta0, 50, L, U, _log_prob, (), scale=1.0,
                         rng=np.random.default_rng(8))
    assert _dentro(pos)


# === initial_positions ===

def test_initial_positions_ball_beats_uniform_in_log_prob():
    """The center of the 'ball' is orders of magnitude better than the mean of 'uniform', which is what
    prevents stuck walkers."""
    pos_b, info_b = initial_positions(64, L, U, _log_prob, (), mode='ball',
                                      seed=9)
    pos_u, info_u = initial_positions(64, L, U, _log_prob, (), mode='uniform',
                                      seed=9)
    assert info_b['mode'] == 'ball' and info_u['mode'] == 'uniform'

    ll_b = np.array([_log_prob(p) for p in pos_b])
    ll_u = np.array([_log_prob(p) for p in pos_u])
    assert np.all(np.isfinite(ll_b))
    assert np.median(ll_b) > np.median(ll_u[np.isfinite(ll_u)] if
                                       np.isfinite(ll_u).any() else [-1e12])
    assert ll_b.std() < 1e3          # ensemble coeso


def test_initial_positions_ball_reports_center_log_prob():
    _pos, info = initial_positions(16, L, U, _log_prob, (), mode='ball', seed=10)
    assert info['center_log_prob'] == pytest.approx(0.0, abs=1.0)  # at the optimum


def test_initial_positions_uniform_has_no_center():
    _pos, info = initial_positions(16, L, U, _log_prob, (), mode='uniform',
                                   seed=11)
    assert info['center_log_prob'] is None


def test_initial_positions_falls_back_to_uniform_when_map_fails():
    def sempre_inf(_t):
        return -np.inf
    pos, info = initial_positions(8, L, U, sempre_inf, (), mode='ball',
                                  n_random=10, seed=12)
    assert 'uniform' in info['mode'] and 'fallback' in info['mode']
    assert pos.shape == (8, 3)


def test_initial_positions_rejects_unknown_mode():
    with pytest.raises(ValueError, match='CALIB_INIT'):
        initial_positions(8, L, U, _log_prob, (), mode='bogus')


def test_initial_positions_is_reproducible_with_seed():
    a, _ = initial_positions(16, L, U, _log_prob, (), mode='ball', seed=13)
    b, _ = initial_positions(16, L, U, _log_prob, (), mode='ball', seed=13)
    np.testing.assert_array_equal(a, b)
