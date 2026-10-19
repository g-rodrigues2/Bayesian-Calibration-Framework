"""Tests of data loading; skipped when the data folder is absent."""
import os

import numpy as np
import pytest

from data_loader import _FNAME_RE, load_control_data, load_data

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
_HAS_DATA = os.path.isdir(_DATA_DIR) and bool(os.listdir(_DATA_DIR))

pytestmark = pytest.mark.skipif(not _HAS_DATA, reason='data folder not found')


def test_regex_parses_canonical_name():
    m = _FNAME_RE.match('control_sensitive_c0_m24_t0_t1_t2_t3_t4_t5')
    assert m is not None
    assert m.group(1) == 'sensitive'
    assert m.group(2) == '0'
    assert m.group(3) == '24'
    assert m.group(4) == '0_t1_t2_t3_t4_t5'


def test_regex_rejects_foreign_name():
    assert _FNAME_RE.match('fig3b_pbs_c3b_m1') is None
    assert _FNAME_RE.match('control_unknownpheno_c0_m1_t0') is None


def test_load_control_data_has_both_groups():
    full = load_control_data(_DATA_DIR)
    assert set(full) == {'control_sensitive', 'control_resistant'}


def test_group_sizes_match_dataset():
    """9 sensitive, 5 resistant."""
    full = load_control_data(_DATA_DIR)
    assert len(full['control_sensitive']) == 9
    assert len(full['control_resistant']) == 5


def test_mouse_names_and_cohorts():
    full = load_control_data(_DATA_DIR)
    resistant = {m['name'] for m in full['control_resistant']}
    assert resistant == {'c1_m2', 'c1_m4', 'c1_m5', 'c1_m7', 'c1_m12'}
    # all resistant mice come from cage c1
    assert {m['cohort'] for m in full['control_resistant']} == {'1'}
    # sensitive mice come from cages c0 and c5
    assert {m['cohort'] for m in full['control_sensitive']} == {'0', '5'}


def test_data_shape_and_treatment_days():
    full = load_control_data(_DATA_DIR)
    for mice in full.values():
        for m in mice:
            assert m['data'].ndim == 2 and m['data'].shape[1] == 2
            assert m['data'][0, 0] == 0.0            # starts at day 0
            assert np.all(np.diff(m['data'][:, 0]) > 0)  # increasing time
            np.testing.assert_array_equal(m['treatment_days'],
                                          [0, 1, 2, 3, 4, 5])
            assert m['treatment'] == 'control'


def test_phenotype_filter():
    full = load_data(_DATA_DIR, phenotypes=['resistant'])
    assert set(full) == {'control_resistant'}
    assert len(full['control_resistant']) == 5


def test_ordering_is_deterministic():
    """The order defines the mapping ic1..icN of the bounds and must be stable."""
    esperado = ['c0_m1', 'c0_m24', 'c0_m29', 'c5_m11', 'c5_m13',
                'c5_m16', 'c5_m27', 'c5_m28', 'c5_m7']
    a = [m['name'] for m in load_control_data(_DATA_DIR)['control_sensitive']]
    b = [m['name'] for m in load_control_data(_DATA_DIR)['control_sensitive']]
    assert a == b == esperado
