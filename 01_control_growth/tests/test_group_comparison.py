"""Tests of the group comparison (box plots and Mann-Whitney U test)."""
import os

import numpy as np
import pytest

from group_comparison import (
    collect_group_posteriors,
    compare_groups,
    mannwhitney,
    plot_group_boxplots,
    run_group_comparison,
)

G1, G2 = 'control_sensitive', 'control_resistant'


# === mannwhitney ===

def test_mannwhitney_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.10, 0.01, 400)
    b = rng.normal(0.30, 0.01, 400)
    u, p = mannwhitney(a, b)
    assert np.isfinite(u)
    assert p < 1e-10


def test_mannwhitney_no_shift_gives_large_p():
    rng = np.random.default_rng(1)
    a = rng.normal(0.1, 0.02, 300)
    b = rng.normal(0.1, 0.02, 300)
    _u, p = mannwhitney(a, b)
    assert p > 0.05


def test_mannwhitney_is_symmetric_in_p():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)
    _u1, p1 = mannwhitney(a, b)
    _u2, p2 = mannwhitney(b, a)
    assert p1 == pytest.approx(p2)


def test_mannwhitney_empty_side():
    u, p = mannwhitney([], [1.0, 2.0])
    assert np.isnan(u) and np.isnan(p)


def test_mannwhitney_identical_constants():
    u, p = mannwhitney(np.full(10, 5.0), np.full(10, 5.0))
    assert np.isnan(u) and p == 1.0


# === helpers that write calibration outputs ===

def _write_global(tmp_path, group, model, chain, labels):
    d = tmp_path / 'Output_Calibration'
    d.mkdir(exist_ok=True, parents=True)
    np.savetxt(d / f'multi_chain_{group}{model}.gz', chain)
    np.savez(d / f'multi_ll_pars_{group}{model}.npz',
             max_ll=-1.0, pars=chain[0], labels=labels)


def _write_mouse(tmp_path, group, mouse, model, chain, labels):
    d = tmp_path / 'Output_Calibration_per_mouse'
    d.mkdir(exist_ok=True, parents=True)
    np.savetxt(d / f'chain_{group}_{mouse}{model}.gz', chain)
    np.savez(d / f'll_pars_{group}_{mouse}{model}.npz',
             max_ll=-1.0, pars=chain[0], labels=labels)


def _chain_log_global(n, r, cc, ics, std=25.0, seed=0):
    """Synthetic _log chain of the global mode: [r, cc, ic1..icN, std]."""
    rng = np.random.default_rng(seed)
    cols = [rng.normal(r, r * 0.02, n), rng.normal(cc, cc * 0.02, n)]
    cols += [rng.normal(ic, 2.0, n) for ic in ics]
    cols.append(rng.normal(std, 1.0, n))
    return np.column_stack(cols)


# === collect_group_posteriors ===

def test_collect_global_mode_maps_labels(tmp_path):
    labels = ['r', 'cc', 'ic1', 'ic2', 'std']
    _write_global(tmp_path, G1, '_log', _chain_log_global(50, 0.1, 2000, [100, 150]),
                  labels)
    out = collect_group_posteriors(str(tmp_path), G1, '_log', mode='global')
    assert set(out) == set(labels)
    assert out['r']['samples'].shape == (50,)
    assert out['r']['best'] == pytest.approx(out['r']['samples'][0])


def test_collect_returns_empty_when_missing(tmp_path):
    assert collect_group_posteriors(str(tmp_path), G1, '_log') == {}


def test_collect_mouse_mode_pools_params_and_keeps_ic_per_mouse(
        tmp_path, synthetic_full_data):
    labels = ['r', 'cc', 'ic', 'std']
    rng = np.random.default_rng(3)
    for mouse in ('c0_m1', 'c0_m2'):
        chain = np.column_stack([rng.normal(0.1, 0.005, 30),
                                 rng.normal(2000, 40, 30),
                                 rng.normal(100, 2, 30),
                                 rng.normal(25, 1, 30)])
        _write_mouse(tmp_path, G1, mouse, '_log', chain, labels)

    out = collect_group_posteriors(str(tmp_path), G1, '_log',
                                   mode='mouse_specific',
                                   full_data=synthetic_full_data)
    # r/cc/std are pooled over the 2 mice; ic has one entry per mouse
    assert out['r']['samples'].shape == (60,)
    assert out['std']['samples'].shape == (60,)
    assert set(k for k in out if k.startswith('ic')) == {'ic[c0_m1]', 'ic[c0_m2]'}
    assert out['ic[c0_m1]']['samples'].shape == (30,)


def test_collect_mouse_mode_requires_full_data(tmp_path):
    with pytest.raises(ValueError, match='full_data'):
        collect_group_posteriors(str(tmp_path), G1, '_log', mode='mouse_specific')


# === compare_groups ===

def test_compare_groups_flags_significant_difference(tmp_path):
    labels1 = ['r', 'cc', 'ic1', 'ic2', 'std']
    labels2 = ['r', 'cc', 'ic1', 'std']
    _write_global(tmp_path, G1, '_log',
                  _chain_log_global(400, 0.10, 2000, [100, 150], seed=4), labels1)
    _write_global(tmp_path, G2, '_log',
                  _chain_log_global(400, 0.30, 2000, [120], seed=5), labels2)

    rows, by_group = compare_groups(str(tmp_path), [G1, G2], '_log')
    assert set(by_group) == {G1, G2}
    pars = {r['param'] for r in rows}
    # initial conditions are excluded from the comparison (ic1 of G1 is not ic1 of G2)
    assert pars == {'r', 'cc', 'std'}

    r_row = next(r for r in rows if r['param'] == 'r')
    assert r_row['significant'] is True          # 0.10 vs 0.30
    assert r_row['median_a'] == pytest.approx(0.10, rel=0.05)
    assert r_row['median_b'] == pytest.approx(0.30, rel=0.05)

    cc_row = next(r for r in rows if r['param'] == 'cc')
    assert cc_row['significant'] is False        # same cc in both groups


def test_compare_groups_needs_two_groups(tmp_path):
    _write_global(tmp_path, G1, '_log',
                  _chain_log_global(20, 0.1, 2000, [100]), ['r', 'cc', 'ic1', 'std'])
    rows, by_group = compare_groups(str(tmp_path), [G1, G2], '_log')
    assert rows == []
    assert set(by_group) == {G1}


def test_compare_groups_std_is_last_param(tmp_path):
    labels = ['r', 'cc', 'ic1', 'std']
    for g, seed in ((G1, 6), (G2, 7)):
        _write_global(tmp_path, g, '_log',
                      _chain_log_global(50, 0.1, 2000, [100], seed=seed), labels)
    rows, _ = compare_groups(str(tmp_path), [G1, G2], '_log')
    assert [r['param'] for r in rows] == ['r', 'cc', 'std']


# === figures and orchestration ===

def test_plot_group_boxplots_writes_png_and_pdf(tmp_path):
    _write_global(tmp_path, G1, '_log',
                  _chain_log_global(80, 0.10, 2000, [100, 150], seed=8),
                  ['r', 'cc', 'ic1', 'ic2', 'std'])
    _write_global(tmp_path, G2, '_log',
                  _chain_log_global(80, 0.30, 2000, [120], seed=9),
                  ['r', 'cc', 'ic1', 'std'])

    rows = plot_group_boxplots(str(tmp_path), [G1, G2], '_log', save=True)
    assert rows
    base = tmp_path / 'Output_Calibration' / 'boxplot_groups_log'
    assert os.path.exists(str(base) + '.png')
    assert os.path.exists(str(base) + '.pdf')


def test_plot_group_boxplots_noop_when_no_chains(tmp_path, capsys):
    assert plot_group_boxplots(str(tmp_path), [G1, G2], '_log') == []
    assert 'no chains' in capsys.readouterr().out


def test_run_group_comparison_writes_csv(tmp_path, synthetic_full_data):
    for g, seed in ((G1, 10), (G2, 11)):
        _write_global(tmp_path, g, '_log',
                      _chain_log_global(60, 0.1 + 0.1 * seed % 2, 2000, [100],
                                        seed=seed),
                      ['r', 'cc', 'ic1', 'std'])

    df = run_group_comparison(str(tmp_path), [G1, G2], models=['_log'],
                              full_data=synthetic_full_data, print_table=False)
    assert not df.empty
    assert set(df['param']) == {'r', 'cc', 'std'}
    assert 'U' in df.columns and 'p' in df.columns and 'significant' in df.columns
    assert os.path.exists(tmp_path / 'Output_Calibration'
                          / 'group_comparison_global.csv')


def test_run_group_comparison_empty(tmp_path, synthetic_full_data):
    df = run_group_comparison(str(tmp_path), [G1, G2], models=['_log'],
                              full_data=synthetic_full_data, print_table=False,
                              save_csv=False, save_fig=False)
    assert df.empty
