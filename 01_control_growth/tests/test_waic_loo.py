"""Tests of WAIC and PSIS-LOO, including parity with arviz."""
import numpy as np
import pytest

from inference import log_likelihood, log_likelihood_pointwise
from model_selection import compute_loo, compute_waic, pointwise_loglik

GROUP = 'control_sensitive'


# === log_likelihood_pointwise ===

def test_pointwise_sums_to_the_scalar_loglik(synthetic_full_data):
    """Invariant: sum_i log p(y_i|theta) == log_likelihood(theta)."""
    theta = np.array([0.1, 100.0, 150.0, 25.0])
    pw = log_likelihood_pointwise(theta, '_exp', synthetic_full_data, GROUP)
    total = log_likelihood(theta, '_exp', synthetic_full_data, GROUP)
    assert pw.shape == (12,)                     # 2 mice x 6 points
    assert pw.sum() == pytest.approx(total, rel=1e-12)


def test_pointwise_mouse_specific_length(synthetic_full_data):
    theta = np.array([0.1, 100.0, 25.0])
    pw = log_likelihood_pointwise(theta, '_exp', synthetic_full_data, GROUP,
                                  mouse_index=0)
    assert pw.shape == (6,)
    assert pw.sum() == pytest.approx(
        log_likelihood(theta, '_exp', synthetic_full_data, GROUP, mouse_index=0))


def test_pointwise_none_on_bad_variance(synthetic_full_data):
    theta = np.array([0.1, 100.0, 150.0, 0.0])   # std = 0
    assert log_likelihood_pointwise(theta, '_exp', synthetic_full_data,
                                    GROUP) is None


def test_pointwise_none_when_ode_fails(monkeypatch, synthetic_full_data):
    import inference

    def boom(*a, **kw):
        raise RuntimeError('falha')

    monkeypatch.setattr(inference, 'solve_model', boom)
    theta = np.array([0.1, 100.0, 150.0, 25.0])
    assert log_likelihood_pointwise(theta, '_exp', synthetic_full_data,
                                    GROUP) is None


def test_pointwise_none_on_nan_solution(monkeypatch, synthetic_full_data):
    import inference

    def nan_solve(*a, **kw):
        out = np.full((6, 1), 100.0)
        out[1, 0] = np.nan
        return out

    monkeypatch.setattr(inference, 'solve_model', nan_solve)
    theta = np.array([0.1, 100.0, 150.0, 25.0])
    assert log_likelihood_pointwise(theta, '_exp', synthetic_full_data,
                                    GROUP) is None


# === pointwise_loglik (matriz) ===

def _chain(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([0.1, 100.0, 150.0, 25.0])
    return base * (1 + rng.normal(0, 5e-3, size=(n, 4)))


def test_pointwise_loglik_matrix_shape(synthetic_full_data):
    m = pointwise_loglik(_chain(40), '_exp', synthetic_full_data, GROUP)
    assert m.shape == (40, 12)
    assert np.all(np.isfinite(m))


def test_pointwise_loglik_subsamples(synthetic_full_data):
    m = pointwise_loglik(_chain(500), '_exp', synthetic_full_data, GROUP,
                         n_sub=50)
    assert m.shape == (50, 12)


def test_pointwise_loglik_rows_sum_to_scalar_loglik(synthetic_full_data):
    ch = _chain(20)
    m = pointwise_loglik(ch, '_exp', synthetic_full_data, GROUP, n_sub=20)
    esperado = [log_likelihood(t, '_exp', synthetic_full_data, GROUP) for t in ch]
    np.testing.assert_allclose(m.sum(axis=1), esperado, rtol=1e-12)


def test_pointwise_loglik_none_when_too_few_valid(monkeypatch,
                                                  synthetic_full_data):
    import model_selection
    monkeypatch.setattr(model_selection, 'log_likelihood_pointwise',
                        lambda *a, **kw: None)
    assert pointwise_loglik(_chain(50), '_exp', synthetic_full_data,
                            GROUP) is None


# === WAIC ===

def test_waic_matches_arviz_up_to_the_ddof_convention():
    """Parity with arviz.waic up to the variance convention.

    Vehtari, Gelman & Gabry (2017), eq. 11, define
        p_waic = sum_i V_{s=1..S}[log p(y_i|theta_s)]
    with V the sample variance (ddof=1); arviz uses ddof=0. The ratio of the two is S/(S-1).
    """
    az = pytest.importorskip('arviz')
    rng = np.random.default_rng(1)
    S, n = 800, 30
    ll = rng.normal(-2.0, 0.3, (S, n))

    waic, p_waic, _frouxos = compute_waic(ll)

    idata = az.from_dict(log_likelihood={'y': ll[None, :, :]})
    ref = az.waic(idata, pointwise=False, scale='deviance')
    fator = S / (S - 1)
    assert p_waic == pytest.approx(float(ref['p_waic']) * fator, rel=1e-10)
    # WAIC = -2 (lppd - p_waic): the difference is 2 (p_waic - p_waic_arviz)
    assert waic == pytest.approx(
        float(ref['waic']) + 2 * (p_waic - float(ref['p_waic'])), rel=1e-10)


def test_waic_p_waic_is_sum_of_pointwise_variances():
    rng = np.random.default_rng(2)
    ll = rng.normal(-1.0, 0.5, (500, 10))
    _w, p_waic, _f = compute_waic(ll)
    assert p_waic == pytest.approx(np.var(ll, axis=0, ddof=1).sum())


def test_waic_zero_variance_gives_zero_penalty():
    """Degenerate posterior (fixed theta): p_waic = 0, WAIC = -2 sum(ll)."""
    ll = np.full((100, 8), -1.5)
    waic, p_waic, n_high_var = compute_waic(ll)
    assert p_waic == pytest.approx(0.0)
    assert n_high_var == 0
    assert waic == pytest.approx(-2 * 8 * -1.5)


def test_waic_flags_loose_points():
    """Var(log p) > 0.4 for some observation increments the counter."""
    rng = np.random.default_rng(3)
    ll = rng.normal(-1.0, 0.05, (500, 6))
    ll[:, 2] = rng.normal(-1.0, 2.0, 500)        # ponto problematico
    _w, _p, n_high_var = compute_waic(ll)
    assert n_high_var == 1


def test_waic_prefers_the_better_model():
    rng = np.random.default_rng(4)
    bom = rng.normal(-1.0, 0.2, (400, 20))
    ruim = rng.normal(-3.0, 0.2, (400, 20))
    assert compute_waic(bom)[0] < compute_waic(ruim)[0]


def test_waic_nan_on_degenerate_input():
    assert np.isnan(compute_waic(np.zeros((5, 3)))[0])
    assert np.isnan(compute_waic(np.zeros(10))[0])


# === PSIS-LOO ===

def test_loo_matches_arviz():
    """Parity with arviz.loo, including p_loo and khat."""
    az = pytest.importorskip('arviz')
    rng = np.random.default_rng(5)
    S, n = 1000, 25
    ll = rng.normal(-2.0, 0.4, (S, n))

    loo, p_loo, khat_max, _pct = compute_loo(ll)

    idata = az.from_dict(log_likelihood={'y': ll[None, :, :]})
    ref = az.loo(idata, pointwise=True, scale='deviance', reff=1.0)
    assert loo == pytest.approx(float(ref['elpd_loo']), rel=1e-10)
    assert p_loo == pytest.approx(float(ref['p_loo']), rel=1e-10)
    assert khat_max == pytest.approx(float(np.max(ref['pareto_k'].values)),
                                     rel=1e-10)


def test_loo_khat_is_low_for_well_behaved_weights():
    rng = np.random.default_rng(6)
    ll = rng.normal(-1.0, 0.15, (2000, 20))      # narrow posterior
    _loo, _p, khat_max, pct = compute_loo(ll)
    assert khat_max < 0.7
    assert pct == 0.0


def test_loo_khat_flags_heavy_tailed_weights():
    """A highly influential observation -> heavy-tailed weights -> high khat."""
    rng = np.random.default_rng(7)
    ll = rng.normal(-1.0, 0.1, (1000, 10))
    # a point explained by few samples: the ratio 1/p is very large
    ll[:, 3] = rng.normal(-1.0, 0.1, 1000)
    ll[:20, 3] = -60.0
    _loo, _p, khat_max, pct = compute_loo(ll)
    assert khat_max > 0.7
    assert pct > 0.0


def test_loo_is_worse_than_waic_penalty_direction():
    """p_loo and p_waic estimate the same quantity and must be close."""
    rng = np.random.default_rng(8)
    ll = rng.normal(-1.5, 0.3, (1500, 30))
    _w, p_waic, _f = compute_waic(ll)
    _l, p_loo, _k, _pct = compute_loo(ll)
    assert p_loo == pytest.approx(p_waic, rel=0.25)


def test_loo_prefers_the_better_model():
    rng = np.random.default_rng(9)
    bom = rng.normal(-1.0, 0.2, (600, 20))
    ruim = rng.normal(-3.0, 0.2, (600, 20))
    assert compute_loo(bom)[0] < compute_loo(ruim)[0]


def test_loo_nan_on_degenerate_input():
    assert np.isnan(compute_loo(np.zeros((5, 3)))[0])
    assert np.isnan(compute_loo(np.zeros(10))[0])


# === integration with the ODE model ===

def test_waic_and_loo_on_real_ode_chain(synthetic_full_data):
    """Complete flow: chain -> pointwise matrix -> finite WAIC and LOO."""
    m = pointwise_loglik(_chain(120, seed=11), '_exp', synthetic_full_data,
                         GROUP, n_sub=120)
    waic, p_waic, _f = compute_waic(m)
    loo, p_loo, khat, _pct = compute_loo(m)
    assert np.isfinite(waic) and np.isfinite(loo)
    assert p_waic > 0 and p_loo > 0
    assert np.isfinite(khat)
    # WAIC and LOO estimate the same quantity and must be close
    assert abs(waic - loo) < 0.15 * abs(waic)
