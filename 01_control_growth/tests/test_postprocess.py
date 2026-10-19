"""Tests of the post-processing functions (bands, summaries, histograms)."""
import os

import numpy as np
import pytest

from postprocess import (
    is_model_param,
    parameter_summary,
    posterior_bands,
    predictive_coverage,
    summarize_posterior,
)


# === is_model_param ===

@pytest.mark.parametrize('label,esperado', [
    ('r', True), ('cc', True), ('b', True), ('c', True),
    ('ic', False), ('ic1', False), ('ic12', False), ('std', False),
    ('IC3', False), ('STD', False),
])
def test_is_model_param(label, esperado):
    assert is_model_param(label) is esperado


# === summarize_posterior ===

def test_summarize_posterior_ordering():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=5.0, scale=1.0, size=20000)
    med, lo, hi, hdi_lo, hdi_hi = summarize_posterior(arr, alpha=0.05)
    assert lo < med < hi
    assert hdi_lo < med < hdi_hi
    assert med == pytest.approx(5.0, abs=0.05)
    # 95% interval of a normal(5, 1) ~ [3.04, 6.96]
    assert lo == pytest.approx(3.04, abs=0.1)
    assert hi == pytest.approx(6.96, abs=0.1)


# === posterior_bands ===

def _fake_chain_exp(n=100, seed=0, std=10.0):
    """Synthetic _exp chain of the mouse-specific mode: [r, ic, std]."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.1, 0.005, n)
    ic = rng.normal(100.0, 2.0, n)
    return np.column_stack([r, ic, np.full(n, std)])


def test_bands_returns_ordered_percentiles():
    chain = _fake_chain_exp()
    t = np.array([0.0, 5.0, 10.0])
    b = posterior_bands(chain, '_exp', t, ndim=1, mouse_idx=0, n_samp=50, seed=1)
    assert b is not None
    assert set(b) == {'t', 'median', 'credible', 'predictive'}
    # smooth grid: dt=0.1 between 0 and 10 -> 101 points
    assert b['t'].shape == b['median'].shape == (101,)
    for key in ('credible', 'predictive'):
        lo, hi = b[key]
        assert lo.shape == hi.shape == (101,)
        assert np.all(lo <= hi)
    np.testing.assert_allclose(b['t'][0], 0.0)
    np.testing.assert_allclose(b['t'][-1], 10.0)


def test_predictive_band_is_wider_than_credible():
    """The predictive band adds observation noise, so it contains the credible band."""
    chain = _fake_chain_exp(n=400, seed=5, std=25.0)
    b = posterior_bands(chain, '_exp', np.array([0.0, 10.0]), ndim=1,
                        mouse_idx=0, n_samp=300, seed=6)
    clo, chi = b['credible']
    plo, phi = b['predictive']
    assert np.all(plo <= clo) and np.all(phi >= chi)
    # and it is strictly wider almost everywhere (std=25 dominates)
    assert np.mean((phi - plo) > (chi - clo)) > 0.99


def test_predictive_width_scales_with_std():
    """Doubling std widens the predictive band but not the credible band."""
    t = np.array([0.0, 5.0])
    b1 = posterior_bands(_fake_chain_exp(n=400, seed=7, std=10.0), '_exp', t,
                         ndim=1, mouse_idx=0, n_samp=400, seed=8)
    b2 = posterior_bands(_fake_chain_exp(n=400, seed=7, std=40.0), '_exp', t,
                         ndim=1, mouse_idx=0, n_samp=400, seed=8)

    cred_w1 = np.mean(b1['credible'][1] - b1['credible'][0])
    cred_w2 = np.mean(b2['credible'][1] - b2['credible'][0])
    pred_w1 = np.mean(b1['predictive'][1] - b1['predictive'][0])
    pred_w2 = np.mean(b2['predictive'][1] - b2['predictive'][0])

    # identical curves (same seed for r/ic) -> unchanged credible band
    assert cred_w2 == pytest.approx(cred_w1, rel=1e-6)
    # the predictive band widens with std
    assert pred_w2 > 2.5 * pred_w1


def test_bands_median_matches_analytic_curve():
    """The median is close to T0 exp(r t) at the median parameter values."""
    chain = _fake_chain_exp(n=500, seed=2)
    b = posterior_bands(chain, '_exp', np.array([0.0, 10.0]), ndim=1,
                        mouse_idx=0, n_samp=500, seed=3)
    esperado = np.median(chain[:, 1]) * np.exp(np.median(chain[:, 0]) * 10)
    assert b['median'][-1] == pytest.approx(esperado, rel=0.02)
    clo, chi = b['credible']
    assert clo[-1] < esperado < chi[-1]


def test_bands_global_mode_picks_right_mouse():
    """Global mode (ndim=2): theta = [r, ic1, ic2, std]; mouse_idx selects the initial condition."""
    n = 200
    chain = np.column_stack([
        np.full(n, 0.1), np.full(n, 100.0), np.full(n, 300.0), np.full(n, 10.0)])
    t = np.array([0.0, 1.0])
    b0 = posterior_bands(chain, '_exp', t, ndim=2, mouse_idx=0, n_samp=10)
    b1 = posterior_bands(chain, '_exp', t, ndim=2, mouse_idx=1, n_samp=10)
    assert b0['median'][0] == pytest.approx(100.0, rel=1e-6)
    assert b1['median'][0] == pytest.approx(300.0, rel=1e-6)


def test_bands_returns_none_when_all_samples_fail(monkeypatch):
    """If every ODE solution fails, None is returned instead of raising."""
    import postprocess

    def bad_solve(*a, **kw):
        raise RuntimeError('falha intencional')

    monkeypatch.setattr(postprocess, 'solve_model', bad_solve)
    out = posterior_bands(_fake_chain_exp(n=10), '_exp', np.array([0.0, 5.0]),
                          ndim=1, mouse_idx=0, n_samp=10)
    assert out is None


def test_bands_skip_nonfinite_samples(monkeypatch):
    """Samples with NaN are skipped and do not affect the percentiles."""
    import postprocess
    real_solve = postprocess.solve_model
    calls = {'n': 0}

    def flaky_solve(*a, **kw):
        calls['n'] += 1
        sol = real_solve(*a, **kw)
        if calls['n'] % 2 == 0:      # half of the samples become NaN
            sol = sol.copy()
            sol[:, 1] = np.nan
        return sol

    monkeypatch.setattr(postprocess, 'solve_model', flaky_solve)
    b = postprocess.posterior_bands(_fake_chain_exp(n=20), '_exp',
                                    np.array([0.0, 5.0]), ndim=1, mouse_idx=0,
                                    n_samp=20)
    assert b is not None
    for key in ('credible', 'predictive'):
        lo, hi = b[key]
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(np.isfinite(b['median']))


def test_mixture_quantile_matches_single_gaussian():
    """With one component, the mixture quantile is the normal quantile mu + z sigma."""
    from postprocess import _mixture_quantile

    f = np.array([[100.0, 200.0]])          # (1, T)
    sigma = np.array([[10.0]])              # (1, 1)
    lo = _mixture_quantile(f, sigma, 0.025)
    hi = _mixture_quantile(f, sigma, 0.975)
    z = 1.959963985
    np.testing.assert_allclose(lo, f[0] - z * 10.0, rtol=1e-5)
    np.testing.assert_allclose(hi, f[0] + z * 10.0, rtol=1e-5)


def test_mixture_quantile_matches_monte_carlo():
    """CDF inversion agrees with sampling the mixture (without Monte Carlo noise)."""
    from postprocess import _mixture_quantile

    rng = np.random.default_rng(21)
    n, T = 60, 5
    f = rng.normal(500.0, 40.0, size=(n, T))
    sigma = rng.uniform(20.0, 60.0, size=(n, 1))

    exato = _mixture_quantile(f, sigma, 0.975)
    draws = f[None] + sigma[None] * rng.normal(size=(4000, n, T))
    mc = np.percentile(draws.reshape(-1, T), 97.5, axis=0)
    np.testing.assert_allclose(exato, mc, rtol=0.02)


def test_predictive_band_is_smooth():
    """The predictive band comes from CDF inversion, not sampling, so it has no Monte Carlo noise: for
    an increasing curve its upper limit is monotone.
    """
    chain = _fake_chain_exp(n=200, seed=19, std=15.0)
    b = posterior_bands(chain, '_exp', np.array([0.0, 10.0]), ndim=1,
                        mouse_idx=0, n_samp=200, seed=20)
    _plo, phi = b['predictive']
    assert np.all(np.diff(phi) > -1e-6)   # non-decreasing


# === predictive_coverage ===

def _mouse_from_curve(t, y):
    return {'name': 'sint', 'data': np.column_stack([t, y])}


def test_predictive_coverage_near_nominal_when_model_is_true():
    """Data generated by the model with N(0, std) noise: coverage ~95%."""
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 10.0, 40)
    verdade = 100.0 * np.exp(0.1 * t)
    y = verdade + rng.normal(0.0, 10.0, t.size)

    chain = _fake_chain_exp(n=800, seed=12, std=10.0)
    b = posterior_bands(chain, '_exp', t, ndim=1, mouse_idx=0, n_samp=500,
                        seed=13)
    frac, k, n = predictive_coverage(b, _mouse_from_curve(t, y))
    assert n == 40 and k == int(frac * n)
    assert 0.85 <= frac <= 1.0     # nominal 0.95, with a margin for sampling noise


def test_predictive_coverage_collapses_when_data_is_shifted():
    """Data shifted away from the curve -> coverage drops."""
    t = np.linspace(0.0, 10.0, 40)
    y = 100.0 * np.exp(0.1 * t) + 500.0     # deslocamento >> std=10
    chain = _fake_chain_exp(n=400, seed=14, std=10.0)
    b = posterior_bands(chain, '_exp', t, ndim=1, mouse_idx=0, n_samp=300, seed=15)
    frac, _k, _n = predictive_coverage(b, _mouse_from_curve(t, y))
    assert frac < 0.1


def test_credible_band_does_not_cover_noisy_data():
    """The credible band covers the curve, not the noisy observations: with a large std it is narrow and
    leaves most points outside, while the predictive band contains them.
    """
    rng = np.random.default_rng(16)
    t = np.linspace(0.0, 10.0, 40)
    y = 100.0 * np.exp(0.1 * t) + rng.normal(0.0, 40.0, t.size)

    chain = _fake_chain_exp(n=600, seed=17, std=40.0)
    b = posterior_bands(chain, '_exp', t, ndim=1, mouse_idx=0, n_samp=400,
                        seed=18)

    j = np.searchsorted(b['t'], t)
    j = np.clip(j, 0, len(b['t']) - 1)
    clo, chi = b['credible']
    credible_coverage = np.mean((y >= clo[j]) & (y <= chi[j]))
    predictive_coverage_frac = predictive_coverage(b, _mouse_from_curve(t, y))[0]

    assert credible_coverage < 0.5          # the credible band does not cover the data
    assert predictive_coverage_frac > 0.85         # the predictive band covers the data


def test_predictive_coverage_none_without_predictive_band():
    b = {'t': np.array([0.0, 1.0]), 'median': np.array([1.0, 2.0]),
         'credible': (np.array([0.0, 1.0]), np.array([2.0, 3.0]))}
    assert predictive_coverage(b, _mouse_from_curve([0.0], [1.0])) is None
    assert predictive_coverage(None, _mouse_from_curve([0.0], [1.0])) is None


# === parameter_summary ===

def test_parameter_summary_defaults_to_model_params(tmp_path):
    """Without params_of_interest only model parameters are summarized (not ic/std)."""
    chain = _fake_chain_exp(n=500)
    path = tmp_path / 'chain.gz'
    np.savetxt(path, chain)

    out = parameter_summary(str(path), labels=['r', 'ic', 'std'])
    assert set(out) == {'r'}
    med, lo, hi, _hdi_lo, _hdi_hi = out['r']
    assert lo < med < hi
    assert med == pytest.approx(0.1, abs=0.002)


def test_parameter_summary_explicit_selection(tmp_path):
    chain = _fake_chain_exp(n=200)
    path = tmp_path / 'chain.gz'
    np.savetxt(path, chain)

    out = parameter_summary(str(path), labels=['r', 'ic', 'std'],
                            params_of_interest=['r', 'ic', 'inexistente'])
    assert set(out) == {'r', 'ic'}   # unknown labels are ignored


# === posterior_histogram ===

def test_plot_posterior_histograms_writes_one_fig_per_model_param(tmp_path):
    from posterior_histogram import plot_posterior_histograms

    n = 300
    rng = np.random.default_rng(4)
    chain = np.column_stack([
        rng.normal(0.2, 0.01, n),      # r
        rng.normal(2000.0, 50.0, n),   # cc
        rng.normal(100.0, 3.0, n),     # ic
        np.full(n, 25.0),              # std
    ])
    chain_path = tmp_path / 'multi_chain_g_log.gz'
    npz_path = tmp_path / 'multi_ll_pars_g_log.npz'
    np.savetxt(chain_path, chain)
    np.savez(npz_path, max_ll=-1.0, pars=chain[0], labels=['r', 'cc', 'ic', 'std'])

    paths = plot_posterior_histograms(
        str(chain_path), npz_path=str(npz_path),
        figure_basename=str(tmp_path / 'hist'))

    # only r and cc -> 2 figures (ic and std are skipped)
    assert len(paths) == 2
    assert all(os.path.exists(p) for p in paths)
    assert os.path.exists(str(tmp_path / 'hist_r.png'))
    assert os.path.exists(str(tmp_path / 'hist_cc.png'))
    assert not os.path.exists(str(tmp_path / 'hist_std.png'))


def test_plot_posterior_histograms_reads_labels_from_npz(tmp_path):
    """labels=None -> read from the .npz file written by global.py."""
    from posterior_histogram import plot_posterior_histograms

    chain = _fake_chain_exp(n=100)
    chain_path = tmp_path / 'chain.gz'
    npz_path = tmp_path / 'pars.npz'
    np.savetxt(chain_path, chain)
    np.savez(npz_path, max_ll=-1.0, pars=chain[0], labels=['r', 'ic', 'std'])

    paths = plot_posterior_histograms(str(chain_path), npz_path=str(npz_path),
                                      figure_basename=str(tmp_path / 'h'))
    assert len(paths) == 1  # only 'r'


def test_plot_posterior_histograms_raises_without_labels(tmp_path):
    from posterior_histogram import plot_posterior_histograms

    chain_path = tmp_path / 'chain.gz'
    np.savetxt(chain_path, _fake_chain_exp(n=10))
    with pytest.raises(ValueError, match='labels'):
        plot_posterior_histograms(str(chain_path),
                                  figure_basename=str(tmp_path / 'h'))
