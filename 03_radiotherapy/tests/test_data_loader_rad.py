"""Tests of the radiotherapy data loading (with dose days)."""
import os

import numpy as np
import pytest

from data_loader_rad import (
    _FNAME_RE, load_data, load_radiation_data, parse_dose_days_env,
)

_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
_HAS = os.path.isdir(_DATA) and bool(os.listdir(_DATA))
pytestmark = pytest.mark.skipif(not _HAS, reason='data/ ausente')


def test_regex_parses_radiation_name():
    m = _FNAME_RE.match('radiation_sensitive_c0_m16_t0_t1_t2_t3_t4_t5')
    assert m.group(1) == 'sensitive'
    assert m.group(2) == '0'
    assert m.group(3) == '16'   # the _t... suffix is ignored


def test_regex_matches_without_suffix():
    """File names do not need the _t... suffix (dose days come from the setting)."""
    m = _FNAME_RE.match('radiation_resistant_c1_m3')
    assert m is not None and m.group(1) == 'resistant'


def test_regex_rejects_control_name():
    assert _FNAME_RE.match('control_sensitive_c0_m1_t0_t1') is None


def test_two_groups_present():
    full = load_radiation_data(_DATA)
    assert set(full) == {'radiation_sensitive', 'radiation_resistant'}


def test_group_sizes():
    full = load_radiation_data(_DATA)
    assert len(full['radiation_sensitive']) == 15
    assert len(full['radiation_resistant']) == 9


def test_default_is_experimental_schedule():
    """Without the setting, the default schedule is the experimental one (days 0-5)."""
    full = load_radiation_data(_DATA)
    for mice in full.values():
        for m in mice:
            np.testing.assert_array_equal(m['treatment_days'], [0, 1, 2, 3, 4, 5])
            assert m['treatment'] == 'radiation'


def test_data_shape():
    full = load_radiation_data(_DATA)
    for mice in full.values():
        for m in mice:
            assert m['data'].ndim == 2 and m['data'].shape[1] == 2
            assert m['data'][0, 0] == 0.0


def test_phenotype_filter():
    full = load_data(_DATA, phenotypes=['resistant'])
    assert set(full) == {'radiation_resistant'}


# === CALIB_DOSE_DAYS: dose schedule independent of the file names ===

def test_dose_days_override_applies_to_all():
    full = load_radiation_data(_DATA, dose_days=[0, 3, 7, 14])
    calendarios = {tuple(m['treatment_days'])
                   for mice in full.values() for m in mice}
    assert calendarios == {(0, 3, 7, 14)}


def test_parse_dose_days_env():
    np.testing.assert_array_equal(parse_dose_days_env('0,1,2,3,4,5'),
                                  [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(parse_dose_days_env('0, 3, 7 , 14'),
                                  [0, 3, 7, 14])
    assert parse_dose_days_env('') is None
    assert parse_dose_days_env(None) is None
    assert parse_dose_days_env('   ') is None
