"""Tests of the detection and removal of stuck walkers."""
import numpy as np
import pytest

from walkers import (
    STUCK_TOL,
    drop_stuck,
    find_stuck_walkers,
    ll_tail,
    walker_mean_loglik,
)


def _log_prob(n_iter, n_good, n_stuck, ll_good=-642.0, ll_stuck=-810.0, seed=0):
    """(n_iter, nwalkers) with `n_good` walkers in the good mode and `n_stuck` in the poor mode."""
    rng = np.random.default_rng(seed)
    a = rng.normal(ll_good, 2.0, (n_iter, n_good))
    b = rng.normal(ll_stuck, 2.0, (n_iter, n_stuck))
    return np.concatenate([a, b], axis=1)


# === ll_tail ===

def test_ll_tail_is_zero_for_symmetric_tight_chain():
    rng = np.random.default_rng(0)
    lp = rng.normal(-642.0, 2.0, 5000)
    assert ll_tail(lp) == pytest.approx(3.3, abs=0.5)   # ~1.64 sigma


def test_ll_tail_catches_lower_tail_that_ll_spread_misses():
    """Observed case: large lower tail but small ll_max - ll_med."""
    lp = _log_prob(1000, n_good=45, n_stuck=3).reshape(-1)
    ll_spread = lp.max() - np.median(lp)      # spread criterion
    tail = ll_tail(lp)                        # lower-tail criterion
    assert ll_spread < 15          # the ll_max - ll_med criterion would not flag it (threshold 50)
    assert tail > 50               # flagged by the lower-tail criterion


def test_ll_tail_handles_nonfinite_and_empty():
    assert ll_tail([-1.0, np.nan, -np.inf, -1.2]) == pytest.approx(0.0, abs=0.3)
    assert np.isnan(ll_tail([]))
    assert np.isnan(ll_tail([np.nan, -np.inf]))


# === walker_mean_loglik ===

def test_walker_mean_loglik_shape_and_values():
    lp = _log_prob(200, n_good=3, n_stuck=2)
    m = walker_mean_loglik(lp)
    assert m.shape == (5,)
    assert np.all(m[:3] > -650) and np.all(m[3:] < -800)


def test_walker_mean_loglik_ignores_nonfinite():
    lp = np.full((10, 2), -100.0)
    lp[0, 0] = -np.inf
    m = walker_mean_loglik(lp)
    assert m[0] == pytest.approx(-100.0)


def test_walker_mean_loglik_rejects_wrong_shape():
    with pytest.raises(ValueError, match='n_iter, nwalkers'):
        walker_mean_loglik(np.zeros(10))


# === find_stuck_walkers ===

def test_find_stuck_walkers_identifies_the_bad_ones():
    lp = _log_prob(500, n_good=45, n_stuck=3)
    n_stuck = find_stuck_walkers(lp)
    assert n_stuck.sum() == 3
    assert np.all(n_stuck[-3:]) and not n_stuck[:-3].any()


def test_find_stuck_walkers_none_when_all_healthy():
    rng = np.random.default_rng(1)
    lp = rng.normal(-642.0, 2.0, (500, 20))
    assert not find_stuck_walkers(lp).any()


def test_find_stuck_walkers_respects_tolerance():
    """A 30-nat difference is not stuck with tol=50 but is with tol=10."""
    lp = _log_prob(300, n_good=10, n_stuck=2, ll_stuck=-672.0)
    assert not find_stuck_walkers(lp, tol=50.0).any()
    assert find_stuck_walkers(lp, tol=10.0).sum() == 2


def test_find_stuck_walkers_refuses_when_majority_is_stuck():
    """With more than 50% flagged there is no reference mode and nothing is removed: the whole chain has
    not converged and removal would hide it."""
    lp = _log_prob(300, n_good=4, n_stuck=6)
    assert not find_stuck_walkers(lp).any()


def test_find_stuck_walkers_marks_nonfinite_walker():
    lp = np.full((100, 4), -100.0)
    lp[:, 2] = -np.inf
    n_stuck = find_stuck_walkers(lp)
    assert n_stuck[2] and n_stuck.sum() == 1


# === drop_stuck ===

def test_drop_stuck_removes_columns_and_preserves_order():
    n_iter, n_good, stuck_n, ndim = 200, 6, 2, 4
    lp = _log_prob(n_iter, n_good, stuck_n)
    rng = np.random.default_rng(2)
    chain = rng.normal(size=(n_iter, n_good + stuck_n, ndim))
    # tag each walker to check that the order is preserved
    chain[:, :, 0] = np.arange(n_good + stuck_n)

    c2, lp2, n_stuck = drop_stuck(chain, lp)
    assert c2.shape == (n_iter, n_good, ndim)
    assert lp2.shape == (n_iter, n_good)
    assert n_stuck.sum() == stuck_n
    np.testing.assert_array_equal(c2[0, :, 0], np.arange(n_good))


def test_drop_stuck_is_noop_when_nothing_is_stuck():
    rng = np.random.default_rng(3)
    lp = rng.normal(-642.0, 2.0, (100, 8))
    chain = rng.normal(size=(100, 8, 3))
    c2, lp2, n_stuck = drop_stuck(chain, lp)
    assert not n_stuck.any()
    np.testing.assert_array_equal(c2, chain)
    np.testing.assert_array_equal(lp2, lp)


def test_drop_stuck_cleans_the_loglik_tail():
    """After removal the lower tail disappears and Var(log L) drops."""
    lp = _log_prob(500, n_good=45, n_stuck=3)
    chain = np.zeros((500, 48, 2))
    var_before = np.var(lp)
    tail_before = ll_tail(lp.reshape(-1))

    _c2, lp2, _stuck = drop_stuck(chain, lp)
    var_after = np.var(lp2)
    tail_after = ll_tail(lp2.reshape(-1))

    assert tail_before > 50 and tail_after < 10
    assert var_after < var_before / 50


def test_stuck_tol_default_is_50():
    assert STUCK_TOL == 50.0
