"""Tests of the inheritance of the growth parameters (independent of the treatment)."""

import numpy as np
import pytest

import growth_params as gpmod

RAD = ['radiation_sensitive', 'radiation_resistant']
IMM = ['immuno_sensitive', 'immuno_resistant']


# === independent of the treatment ===

def test_phenotype_of():
    assert gpmod.phenotype_of('radiation_sensitive') == 'sensitive'
    assert gpmod.phenotype_of('immuno_resistant') == 'resistant'
    assert gpmod.phenotype_of('qualquercoisa_sensitive') == 'sensitive'


def test_fixed_source_radiotherapy():
    g = gpmod.load_growth_params(RAD, source='fixed', verbose=False)
    assert set(g) == set(RAD)
    assert g['radiation_sensitive']['cc'] == g['radiation_resistant']['cc']


def test_fixed_source_immunotherapy_same_module():
    """The same module serves immunotherapy groups."""
    g = gpmod.load_growth_params(IMM, source='fixed', verbose=False)
    assert set(g) == set(IMM)
    # immunotherapy and radiotherapy groups of the same phenotype inherit the same values
    r = gpmod.load_growth_params(RAD, source='fixed', verbose=False)
    assert g['immuno_sensitive']['r'] == r['radiation_sensitive']['r']


# === selected control model ===

def _write_control(dir_, group, model, max_ll, k):
    d = dir_ / 'Output_Calibration'
    d.mkdir(exist_ok=True, parents=True)
    labels = ['r', 'cc'] + [f'ic{i}' for i in range(k - 3)] + ['std']
    pars = np.array([0.1, 3800.0] + [100.0] * (k - 3) + [30.0])
    np.savez(d / f'multi_ll_pars_{group}{model}.npz', max_ll=max_ll,
             pars=pars, labels=np.array(labels))


def _write_data_stub(dir_):
    """Control data with one file per phenotype (for the number of observations)."""
    d = dir_ / 'data'
    d.mkdir(exist_ok=True, parents=True)
    for tag, c in [('sensitive', '0'), ('resistant', '1')]:
        np.savetxt(d / f'control_{tag}_c{c}_m1_t0_t1.txt',
                   np.column_stack((np.arange(10.), np.arange(10.) + 100)))


def test_discover_control_model_picks_best_bic(tmp_path, monkeypatch):
    _write_data_stub(tmp_path)
    # _log has a better max_ll than _exp with the same k -> lower BIC
    for group in ('control_sensitive', 'control_resistant'):
        _write_control(tmp_path, group, '_log', -50.0, 4)
        _write_control(tmp_path, group, '_exp', -90.0, 3)
    monkeypatch.setattr(gpmod, 'CONTROL_DIR', str(tmp_path))
    monkeypatch.setattr(gpmod, 'DATA_DIR', str(tmp_path / 'data'))
    assert gpmod.discover_control_model('bic') == '_log'


def test_discover_control_model_none_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(gpmod, 'CONTROL_DIR', str(tmp_path))
    assert gpmod.discover_control_model('bic') is None


# === selected sharing scenario ===

def _write_scenario(dir_, model, snum, max_ll, labels, pars):
    d = dir_ / 'Output_Calibration'
    d.mkdir(exist_ok=True, parents=True)
    np.savez(d / f'multi_ll_pars_{model}_s{snum}.npz', max_ll=max_ll,
             pars=np.array(pars), labels=np.array(labels))


def test_discover_shared_scenario_picks_best(tmp_path, monkeypatch):
    _write_data_stub(tmp_path)
    monkeypatch.setattr(gpmod, 'CONTROL_DIR', str(tmp_path))
    monkeypatch.setattr(gpmod, 'SHARED_DIR', str(tmp_path))
    # s2 (smaller k, same max_ll) -> better BIC than s1
    _write_scenario(tmp_path, '_log', 1, -50.0,
                    ['r_sensitive', 'r_resistant', 'cc_sensitive', 'cc_resistant', 'std'],
                    [0.08, 0.11, 4400, 3700, 30])
    _write_scenario(tmp_path, '_log', 2, -50.0,
                    ['r_sensitive', 'r_resistant', 'cc_global', 'std'],
                    [0.08, 0.11, 3800, 30])
    assert gpmod.discover_shared_scenario('_log', 'bic') == 's2'


# === loading from a scenario (generic labels) ===

def test_from_shared_scenario_s2_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(gpmod, 'SHARED_DIR', str(tmp_path))
    _write_scenario(tmp_path, '_log', 2, -50.0,
                    ['r_sensitive', 'r_resistant', 'cc_global', 'ic1', 'std'],
                    [0.08, 0.12, 3900, 100, 30])
    g = gpmod.load_growth_params(RAD, source='shared:s2', verbose=False)
    assert g['radiation_sensitive']['r'] == pytest.approx(0.08)
    assert g['radiation_resistant']['r'] == pytest.approx(0.12)
    # cc_global -> same cc for both groups
    assert g['radiation_sensitive']['cc'] == g['radiation_resistant']['cc'] == 3900


def test_shared_source_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(gpmod, 'SHARED_DIR', str(tmp_path))
    with pytest.raises(FileNotFoundError):
        gpmod.load_growth_params(RAD, source='shared:s2', verbose=False)


# === auto: selection and loading, with fallback ===

def test_auto_prefers_shared_winner(tmp_path, monkeypatch):
    _write_data_stub(tmp_path)
    monkeypatch.setattr(gpmod, 'CONTROL_DIR', str(tmp_path))
    monkeypatch.setattr(gpmod, 'SHARED_DIR', str(tmp_path))
    for group in ('control_sensitive', 'control_resistant'):
        _write_control(tmp_path, group, '_log', -50.0, 4)
    _write_scenario(tmp_path, '_log', 2, -40.0,
                    ['r_sensitive', 'r_resistant', 'cc_global', 'std'],
                    [0.07, 0.13, 3600, 30])
    g = gpmod.load_growth_params(RAD, source='auto', verbose=False)
    assert g['radiation_sensitive']['r'] == pytest.approx(0.07)
    assert g['radiation_resistant']['cc'] == 3600


def test_auto_falls_back_to_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(gpmod, 'CONTROL_DIR', str(tmp_path / 'a'))
    monkeypatch.setattr(gpmod, 'SHARED_DIR', str(tmp_path / 'b'))
    g = gpmod.load_growth_params(RAD, source='auto', verbose=False)
    assert g == gpmod._fixed_for(RAD)
