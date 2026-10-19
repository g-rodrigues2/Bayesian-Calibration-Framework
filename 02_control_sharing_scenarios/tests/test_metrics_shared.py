"""Tests of scenario selection (compare_scenarios)."""

import numpy as np
import pytest

from metrics_shared import (
    compare_scenarios, n_observations, print_comparison_table,
)
from scenarios import SCENARIOS

G = ['control_sensitive', 'control_resistant']


def test_n_observations(synthetic_two_groups):
    assert n_observations(synthetic_two_groups, G) == 24   # 4 mice x 6 points


def _write_scenario(tmp_path, model, snum, max_ll, theta, chain=None):
    d = tmp_path / 'Output_Calibration'
    d.mkdir(exist_ok=True)
    stem = f'{model}_s{snum}'
    np.savez(d / f'multi_ll_pars_{stem}.npz', max_ll=max_ll,
             pars=np.asarray(theta), labels=np.arange(len(theta)))
    if chain is not None:
        np.savetxt(d / f'multi_chain_{stem}.gz', chain)


def test_compare_scenarios_bic_only(tmp_path, synthetic_two_groups):
    # s4 (all global, smaller k) versus s1 (nothing shared, larger k), same max_ll
    _write_scenario(tmp_path, '_log', 4, -100.0, np.ones(7))   # k=7
    _write_scenario(tmp_path, '_log', 1, -100.0, np.ones(9))   # k=9
    df = compare_scenarios(str(tmp_path / 'Output_Calibration'),
                           synthetic_two_groups, G, '_log',
                           SCENARIOS['_log'], metrics=['bic'])
    # BIC penalizes k: s4 wins with the same max_ll
    assert df.iloc[0]['scenario'] == 's4'
    assert df.iloc[0]['dBIC'] == 0.0
    assert df['w_BIC'].sum() == pytest.approx(1.0)
    assert list(df.columns) == ['scenario', 'model', 'k', 'n', 'max_ll',
                                'BIC', 'dBIC', 'w_BIC']


def test_compare_scenarios_rewards_better_fit(tmp_path, synthetic_two_groups):
    # same k, better max_ll -> lower BIC
    _write_scenario(tmp_path, '_exp', 4, -90.0, np.ones(6))
    _write_scenario(tmp_path, '_exp', 1, -80.0, np.ones(6))   # better fit
    df = compare_scenarios(str(tmp_path / 'Output_Calibration'),
                           synthetic_two_groups, G, '_exp',
                           SCENARIOS['_exp'], metrics=['bic'])
    assert df.iloc[0]['scenario'] == 's1'   # better max_ll with the same k


def test_compare_scenarios_skips_missing(tmp_path, synthetic_two_groups):
    _write_scenario(tmp_path, '_log', 4, -100.0, np.ones(7))
    df = compare_scenarios(str(tmp_path / 'Output_Calibration'),
                           synthetic_two_groups, G, '_log',
                           SCENARIOS['_log'], metrics=['bic'])
    assert set(df['scenario']) == {'s4'}


def test_compare_scenarios_empty(tmp_path, synthetic_two_groups):
    df = compare_scenarios(str(tmp_path / 'Output_Calibration'),
                           synthetic_two_groups, G, '_log',
                           SCENARIOS['_log'], metrics=['bic'])
    assert df.empty


def test_compare_scenarios_full_metrics_with_chain(tmp_path, synthetic_two_groups):
    """With a chain, DIC/WAIC/LOO are also computed."""
    from bounds_shared import define_shared_bounds_labels
    sdef = SCENARIOS['_exp'][4]
    l, u, _lab = define_shared_bounds_labels('_exp', sdef, G,
                                             synthetic_two_groups, max_ic=200.0)
    rng = np.random.default_rng(0)
    theta0 = np.array([0.13, 100., 50., 80., 120., 5.0])
    chain = theta0 * (1 + rng.normal(0, 1e-3, size=(60, 6)))
    _write_scenario(tmp_path, '_exp', 4, -50.0, theta0, chain=chain)

    df = compare_scenarios(str(tmp_path / 'Output_Calibration'),
                           synthetic_two_groups, G, '_exp',
                           {4: sdef}, metrics=['aicc', 'bic', 'dic', 'waic', 'loo'])
    r = df.iloc[0]
    assert np.isfinite(r['AICc']) and np.isfinite(r['BIC'])
    assert np.isfinite(r['DIC']) and np.isfinite(r['p_D'])
    assert np.isfinite(r['WAIC']) and np.isfinite(r['LOO'])
    assert np.isfinite(r['khat_max'])


def test_print_comparison_table_smoke(tmp_path, synthetic_two_groups, capsys):
    _write_scenario(tmp_path, '_log', 4, -100.0, np.ones(7))
    _write_scenario(tmp_path, '_log', 2, -95.0, np.ones(8))
    df = compare_scenarios(str(tmp_path / 'Output_Calibration'),
                           synthetic_two_groups, G, '_log',
                           SCENARIOS['_log'], metrics=['bic'])
    print_comparison_table(df, '_log', metrics=['bic'])
    out = capsys.readouterr().out
    assert 'Parameter sharing: _log' in out
    assert 'best BIC' in out
