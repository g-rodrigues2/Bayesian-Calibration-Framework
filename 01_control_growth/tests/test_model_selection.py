"""Tests of model selection (information criteria, deltas and weights)."""
import os

import numpy as np
import pytest

from model_selection import (
    ALL_METRICS,
    bic_weights,
    compare_models,
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_dic,
    compute_wbic,
    load_chain_and_loglik,
    n_observations,
    normalize_metrics,
    run_model_selection,
)

GROUP = 'control_sensitive'   # 2 mice with 6 points each in the fixture


# === AIC / AICc ===

def test_aic_formula():
    assert compute_aic(-100.0, 4) == pytest.approx(2 * 4 - 2 * (-100.0))


def test_aic_penalizes_less_than_bic_for_large_n():
    """2k < k ln(n) whenever n > e^2 ~ 7.39, so BIC penalizes more."""
    max_ll, k, n = -100.0, 5, 105
    assert compute_aic(max_ll, k) < compute_bic(max_ll, k, n)


def test_aicc_adds_positive_correction():
    max_ll, k, n = -100.0, 5, 30
    corr = 2 * k * (k + 1) / (n - k - 1)
    assert compute_aicc(max_ll, k, n) == pytest.approx(compute_aic(max_ll, k) + corr)
    assert corr > 0


def test_aicc_converges_to_aic_for_large_n():
    max_ll, k = -100.0, 5
    assert compute_aicc(max_ll, k, 10**7) == pytest.approx(compute_aic(max_ll, k),
                                                           abs=1e-3)


def test_aicc_nan_when_correction_undefined():
    """n - k - 1 <= 0 -> undefined correction; NaN instead of -inf."""
    assert np.isnan(compute_aicc(-10.0, 5, 6))
    assert np.isnan(compute_aicc(-10.0, 5, 3))


def test_aicc_penalizes_more_than_aic_at_small_n():
    """With small n/k, AICc penalizes complex models more."""
    n = 105
    d_aic = compute_aic(-100.0, 13) - compute_aic(-100.0, 7)
    d_aicc = compute_aicc(-100.0, 13, n) - compute_aicc(-100.0, 7, n)
    assert d_aicc > d_aic


# === normalize_metrics ===

def test_normalize_metrics_defaults_to_all():
    assert normalize_metrics(None) == list(ALL_METRICS)


def test_normalize_metrics_accepts_string_and_canonical_order():
    assert normalize_metrics('wbic,aic') == ['aic', 'wbic']
    assert normalize_metrics(['BIC', 'aic']) == ['aic', 'bic']


def test_normalize_metrics_dedupes():
    assert normalize_metrics('bic,bic') == ['bic']


def test_normalize_metrics_rejects_unknown():
    with pytest.raises(ValueError, match='Unknown metric'):
        normalize_metrics('bic,cic')


def test_normalize_metrics_rejects_empty():
    with pytest.raises(ValueError, match='at least one'):
        normalize_metrics([])


# === n_observations ===

def test_n_observations_group_and_mouse(synthetic_full_data):
    assert n_observations(synthetic_full_data, GROUP) == 12          # 2 x 6
    assert n_observations(synthetic_full_data, GROUP, mouse_index=0) == 6
    assert n_observations(synthetic_full_data, 'control_resistant') == 6


# === BIC ===

def test_bic_matches_schwarz_formula():
    """Formula of the original notebook: k log(n) - 2 max_ll."""
    max_ll, k, n = -638.9, 8, 105
    assert compute_bic(max_ll, k, n) == pytest.approx(k * np.log(n) - 2 * max_ll)


def test_bic_penalizes_extra_parameters():
    assert compute_bic(-100.0, 5, 50) > compute_bic(-100.0, 3, 50)


def test_bic_rewards_better_fit():
    assert compute_bic(-90.0, 3, 50) < compute_bic(-100.0, 3, 50)


def test_bic_nan_when_no_observations():
    assert np.isnan(compute_bic(-10.0, 2, 0))


# === Schwarz weights ===

def test_bic_weights_sum_to_one_and_favor_minimum():
    w = bic_weights([100.0, 104.0, 110.0])
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
    esperado = np.exp([-0.0, -2.0, -5.0])       # dBIC = [0, 4, 10]
    np.testing.assert_allclose(w, esperado / esperado.sum(), rtol=1e-10)


def test_bic_weights_identical_models_split_evenly():
    np.testing.assert_allclose(bic_weights([50.0, 50.0]), [0.5, 0.5])


def test_bic_weights_ignore_nan():
    w = bic_weights([100.0, np.nan, 104.0])
    assert w[1] == 0.0
    assert w[0] + w[2] == pytest.approx(1.0)


def test_bic_weights_all_nan():
    assert np.all(np.isnan(bic_weights([np.nan, np.nan])))


def test_bic_weights_dominant_model_saturates_without_overflow():
    w = bic_weights([0.0, 500.0])
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.0, abs=1e-12)


# === DIC ===

def test_dic_identity_dbar_plus_pd():
    """DIC = D(theta_bar) + 2 p_D = D_bar + p_D."""
    ll = np.full(100, -50.0)
    ll[0] = -52.0
    ll_at_mean = -49.0
    d, p_D = compute_dic(ll, ll_at_mean)
    D_bar = float((-2.0 * ll).mean())
    D_at_mean = -2.0 * ll_at_mean
    assert p_D == pytest.approx(D_bar - D_at_mean)
    assert d == pytest.approx(D_at_mean + 2 * p_D)
    assert d == pytest.approx(D_bar + p_D)


def test_dic_pd_positive_for_spread_posterior():
    """Mean log_lik worse than at the mean theta -> p_D > 0 (regular case)."""
    rng = np.random.default_rng(0)
    ll = rng.normal(-60.0, 2.0, 500)
    d, p_D = compute_dic(ll, log_lik_at_mean=-55.0)
    assert p_D > 0 and np.isfinite(d)


def test_dic_pd_negative_flags_multimodality():
    """If the mean theta fits worse than the average sample, p_D < 0."""
    ll = np.full(200, -50.0)
    _d, p_D = compute_dic(ll, log_lik_at_mean=-500.0)  # theta_bar between modes
    assert p_D < 0


def test_dic_nan_when_too_few_samples():
    d, p_D = compute_dic(np.full(5, -10.0), -10.0)
    assert np.isnan(d) and np.isnan(p_D)


def test_dic_nan_when_mean_is_not_finite():
    d, p_D = compute_dic(np.full(50, -10.0), -np.inf)
    assert np.isnan(d) and np.isnan(p_D)


def test_dic_ignores_nonfinite_samples():
    ll = np.concatenate([np.full(50, -10.0), np.full(10, -np.inf)])
    d, p_D = compute_dic(ll, -9.0)
    assert np.isfinite(d) and np.isfinite(p_D)


# === WBIC ===

def test_wbic_equals_minus_2_mean_ll_when_beta_is_one():
    """With n = e, beta* = 1: uniform weights -> WBIC = -2 mean(ll)."""
    rng = np.random.default_rng(1)
    ll = rng.normal(-40.0, 1.0, 500)
    w, _ess = compute_wbic(ll, n=np.e)
    assert w == pytest.approx(-2.0 * ll.mean(), rel=1e-9)


def test_wbic_importance_weights_match_manual_formula():
    rng = np.random.default_rng(2)
    ll = rng.normal(-100.0, 5.0, 300)
    n = 120
    w, ess = compute_wbic(ll, n)

    beta = 1.0 / np.log(n)
    lw = (beta - 1.0) * ll
    lw -= lw.max()
    ww = np.exp(lw)
    assert w == pytest.approx(-2.0 * (ww * ll).sum() / ww.sum(), rel=1e-10)
    assert ess == pytest.approx(ww.sum() ** 2 / (ww ** 2).sum(), rel=1e-10)


def test_wbic_reweights_toward_lower_likelihood():
    """beta* < 1 flattens the likelihood, so the weighted mean of ll is below the plain mean and
    WBIC > -2 mean(ll)."""
    rng = np.random.default_rng(3)
    ll = rng.normal(-100.0, 5.0, 500)
    w, _ = compute_wbic(ll, n=1000)
    assert w > -2.0 * ll.mean()


def test_wbic_ess_degrades_with_spread():
    """A large spread of log_lik concentrates the weights and lowers the ESS."""
    rng = np.random.default_rng(4)
    _w1, ess_narrow = compute_wbic(rng.normal(-50, 0.5, 500), n=100)
    _w2, ess_wide = compute_wbic(rng.normal(-50, 30.0, 500), n=100)
    assert ess_narrow > ess_wide


def test_wbic_nan_when_too_few_samples():
    w, ess = compute_wbic(np.full(5, -10.0), n=100)
    assert np.isnan(w) and np.isnan(ess)


def test_wbic_is_not_bic_weights():
    """BIC weights must not be confused with WBIC."""
    ll = np.full(100, -50.0)
    w, _ = compute_wbic(ll, n=50)
    assert w == pytest.approx(100.0)         # -2 * (-50)
    weights = bic_weights([100.0, 104.0])      # a different quantity
    assert weights.sum() == pytest.approx(1.0)
    assert not np.isclose(w, weights[0])


# === load_chain_and_loglik ===

def test_load_chain_falls_back_to_gz_and_recomputes_loglik(tmp_path,
                                                           synthetic_full_data):
    """Without the .h5 backend, the .gz chain is read and log_lik is recomputed."""
    theta = np.array([0.1, 100.0, 150.0, 10.0])
    chain = np.tile(theta, (30, 1))
    gz = tmp_path / 'chain.gz'
    np.savetxt(gz, chain)

    samples, log_lik = load_chain_and_loglik(
        str(tmp_path / 'ausente.h5'), str(gz), '_exp', synthetic_full_data, GROUP)
    assert samples.shape == (30, 4)
    assert len(log_lik) == 30
    assert np.all(np.isfinite(log_lik))


def test_load_chain_returns_none_when_nothing_exists(tmp_path, synthetic_full_data):
    s, ll = load_chain_and_loglik(str(tmp_path / 'a.h5'), str(tmp_path / 'b.gz'),
                                  '_exp', synthetic_full_data, GROUP)
    assert s is None and ll is None


# === compare_models / run_model_selection ===

def _write(tmp_path, mode, group, model, max_ll, theta, chain=None, mouse='c0_m1'):
    if mode == 'global':
        d = tmp_path / 'Output_Calibration'
        npz, gz = (f'multi_ll_pars_{group}{model}.npz',
                   f'multi_chain_{group}{model}.gz')
    else:
        d = tmp_path / 'Output_Calibration_per_mouse'
        npz, gz = (f'll_pars_{group}_{mouse}{model}.npz',
                   f'chain_{group}_{mouse}{model}.gz')
    d.mkdir(exist_ok=True, parents=True)
    np.savez(d / npz, max_ll=max_ll, pars=np.asarray(theta))
    if chain is not None:
        np.savetxt(d / gz, chain)


def test_compare_models_bic_only_omits_other_metric_keys(tmp_path,
                                                         synthetic_full_data):
    """metrics=['bic'] creates no DIC/WBIC columns and does not read the chain."""
    _write(tmp_path, 'global', GROUP, '_exp', -50.0, [0.1, 100.0, 150.0, 10.0])
    rows = compare_models(str(tmp_path), synthetic_full_data, GROUP,
                          mode='global', models=['_exp'], metrics=['bic'])
    r = rows[0]
    assert r['n_obs'] == 12 and r['k'] == 4
    assert np.isfinite(r['BIC'])
    assert 'DIC' not in r and 'WBIC' not in r and 'AIC' not in r
    assert r['n_samples'] == 0            # no chain was read


def test_compare_models_maxll_metrics_never_read_the_chain(tmp_path,
                                                           synthetic_full_data,
                                                           monkeypatch):
    """aic/aicc/bic depend only on max_ll: load_chain_and_loglik is not called."""
    import model_selection

    def boom(*a, **kw):
        raise AssertionError('the chain must not be read')

    monkeypatch.setattr(model_selection, 'load_chain_and_loglik', boom)
    _write(tmp_path, 'global', GROUP, '_exp', -50.0, [0.1, 100.0, 150.0, 10.0])
    rows = model_selection.compare_models(
        str(tmp_path), synthetic_full_data, GROUP, mode='global',
        models=['_exp'], metrics=['aic', 'aicc', 'bic'])
    assert set(['AIC', 'AICc', 'BIC']).issubset(rows[0])


def test_compare_models_wbic_only_skips_dic_theta_bar(tmp_path,
                                                      synthetic_full_data):
    """Requesting only wbic, DIC (which solves the ODE at theta_bar) is absent."""
    theta = np.array([0.1, 100.0, 150.0, 10.0])
    rng = np.random.default_rng(9)
    chain = theta * (1 + rng.normal(0, 1e-3, size=(40, 4)))
    _write(tmp_path, 'global', GROUP, '_exp', -50.0, theta, chain=chain)
    rows = compare_models(str(tmp_path), synthetic_full_data, GROUP,
                          mode='global', models=['_exp'], metrics=['wbic'])
    assert np.isfinite(rows[0]['WBIC'])
    assert 'DIC' not in rows[0] and 'p_D' not in rows[0]


def test_compare_models_bic_matches_notebook_formula(tmp_path, synthetic_full_data):
    theta, max_ll = [0.1, 100.0, 150.0, 10.0], -123.456
    _write(tmp_path, 'global', GROUP, '_exp', max_ll, theta)
    rows = compare_models(str(tmp_path), synthetic_full_data, GROUP,
                          mode='global', models=['_exp'], metrics=['bic'])
    assert rows[0]['BIC'] == pytest.approx(len(theta) * np.log(12) - 2 * max_ll)


def test_compare_models_computes_dic_and_wbic_from_gz(tmp_path, synthetic_full_data):
    theta = np.array([0.1, 100.0, 150.0, 10.0])
    rng = np.random.default_rng(5)
    chain = theta * (1 + rng.normal(0, 1e-3, size=(40, 4)))
    _write(tmp_path, 'global', GROUP, '_exp', -50.0, theta, chain=chain)

    rows = compare_models(str(tmp_path), synthetic_full_data, GROUP,
                          mode='global', models=['_exp'], metrics=['bic','dic','wbic'])
    r = rows[0]
    assert r['n_samples'] == 40
    assert np.isfinite(r['DIC']) and np.isfinite(r['p_D'])
    assert np.isfinite(r['WBIC']) and np.isfinite(r['WBIC_ESS'])


def test_compare_models_skips_missing_npz(tmp_path, synthetic_full_data):
    _write(tmp_path, 'global', GROUP, '_exp', -50.0, [0.1, 100.0, 150.0, 10.0])
    rows = compare_models(str(tmp_path), synthetic_full_data, GROUP,
                          mode='global', models=['_exp', '_log', '_gomp'],
                          metrics=['bic'])
    assert [r['model'] for r in rows] == ['_exp']


def test_compare_models_empty_when_nothing_calibrated(tmp_path, synthetic_full_data):
    assert compare_models(str(tmp_path), synthetic_full_data, GROUP,
                          mode='global', metrics=['bic']) == []


def test_compare_models_mouse_mode_uses_mouse_n(tmp_path, synthetic_full_data):
    _write(tmp_path, 'mouse_specific', GROUP, '_exp', -20.0, [0.1, 100.0, 10.0])
    rows = compare_models(str(tmp_path), synthetic_full_data, GROUP,
                          mode='mouse_specific', models=['_exp'],
                          mouse_index=0, metrics=['bic'])
    assert len(rows) == 1
    assert rows[0]['n_obs'] == 6          # only the points of that mouse
    assert rows[0]['k'] == 3
    assert rows[0]['mouse'] == 'c0_m1'


def test_run_model_selection_columns_weights_and_csv(tmp_path, synthetic_full_data):
    for g in ('control_sensitive', 'control_resistant'):
        _write(tmp_path, 'global', g, '_exp', -50.0, [0.1, 100.0, 150.0, 10.0])
        _write(tmp_path, 'global', g, '_log', -49.0,
               [0.1, 2000.0, 100.0, 150.0, 10.0])

    df = run_model_selection(results_dir=str(tmp_path), mode='global',
                             full_data=synthetic_full_data,
                             metrics=['bic'], print_table=False)

    # only the BIC columns
    assert list(df.columns) == [
        'group', 'mouse', 'model', 'n_obs', 'k', 'n_samples', 'max_ll',
        'BIC', 'dBIC', 'w_BIC']
    assert len(df) == 4
    # weights sum to 1 and the best model has dBIC = 0 within each group
    for _g, sub in df.groupby('group'):
        assert sub['w_BIC'].sum() == pytest.approx(1.0)
        assert sub['dBIC'].min() == pytest.approx(0.0)
    assert os.path.exists(tmp_path / 'Output_Calibration'
                          / 'model_selection_global.csv')


def test_run_model_selection_column_order_follows_metric_order(
        tmp_path, synthetic_full_data):
    """Columns follow the canonical order aic, aicc, bic, dic, wbic."""
    _write(tmp_path, GROUP_MODE := 'global', GROUP, '_exp', -50.0,
           [0.1, 100.0, 150.0, 10.0])
    df = run_model_selection(results_dir=str(tmp_path), mode=GROUP_MODE,
                             groups=[GROUP], models=['_exp'],
                             full_data=synthetic_full_data,
                             metrics='bic,aicc,aic', print_table=False,
                             save_csv=False)
    assert list(df.columns) == [
        'group', 'mouse', 'model', 'n_obs', 'k', 'n_samples', 'max_ll',
        'AIC', 'dAIC', 'w_AIC', 'AICc', 'dAICc', 'w_AICc',
        'BIC', 'dBIC', 'w_BIC']


def test_run_model_selection_all_metrics_present(tmp_path, synthetic_full_data):
    theta = np.array([0.1, 100.0, 150.0, 10.0])
    rng = np.random.default_rng(12)
    chain = theta * (1 + rng.normal(0, 1e-3, size=(40, 4)))
    _write(tmp_path, 'global', GROUP, '_exp', -50.0, theta, chain=chain)
    df = run_model_selection(results_dir=str(tmp_path), mode='global',
                             groups=[GROUP], models=['_exp'],
                             full_data=synthetic_full_data, metrics=None,
                             print_table=False, save_csv=False)
    for c in ('AIC', 'AICc', 'BIC', 'DIC', 'p_D', 'WBIC', 'WBIC_ESS'):
        assert c in df.columns


def test_run_model_selection_rejects_bad_metric(tmp_path, synthetic_full_data):
    with pytest.raises(ValueError, match='Unknown metric'):
        run_model_selection(results_dir=str(tmp_path), mode='global',
                            full_data=synthetic_full_data, metrics='bogus',
                            print_table=False, save_csv=False)


def test_run_model_selection_deltas_are_per_mouse(tmp_path, synthetic_full_data):
    """In the mouse-specific mode each mouse has its own best model."""
    _write(tmp_path, 'mouse_specific', GROUP, '_exp', -20.0, [0.1, 100.0, 10.0],
           mouse='c0_m1')
    _write(tmp_path, 'mouse_specific', GROUP, '_log', -30.0,
           [0.1, 2000.0, 100.0, 10.0], mouse='c0_m1')
    _write(tmp_path, 'mouse_specific', GROUP, '_exp', -40.0, [0.1, 150.0, 10.0],
           mouse='c0_m2')
    _write(tmp_path, 'mouse_specific', GROUP, '_log', -20.0,
           [0.1, 2000.0, 150.0, 10.0], mouse='c0_m2')

    df = run_model_selection(results_dir=str(tmp_path), mode='mouse_specific',
                             groups=[GROUP], models=['_exp', '_log'],
                             full_data=synthetic_full_data,
                             metrics=['bic'], print_table=False,
                             save_csv=False)
    assert set(df['mouse']) == {'c0_m1', 'c0_m2'}
    for _m, sub in df.groupby('mouse'):
        assert sub['dBIC'].min() == pytest.approx(0.0)
        assert sub['w_BIC'].sum() == pytest.approx(1.0)
    # m1 prefers _exp (better ll and smaller k); m2 prefers _log
    assert df[df['mouse'] == 'c0_m1'].sort_values('BIC').iloc[0]['model'] == '_exp'
    assert df[df['mouse'] == 'c0_m2'].sort_values('BIC').iloc[0]['model'] == '_log'


def test_run_model_selection_empty_returns_empty_df(tmp_path, synthetic_full_data):
    df = run_model_selection(results_dir=str(tmp_path), mode='global',
                             full_data=synthetic_full_data,
                             metrics=['bic'], print_table=False,
                             save_csv=False)
    assert df.empty
