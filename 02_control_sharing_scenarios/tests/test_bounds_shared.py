"""Tests of the bounds and labels of the multi-group theta."""

from bounds_shared import define_shared_bounds_labels
from scenarios import SCENARIOS

G = ['control_sensitive', 'control_resistant']


def test_exp_s1_group_specific(synthetic_two_groups):
    # [r_sens, r_res, T0(2 sens), T0(2 res), std] = 2+2+2+1 = 7
    l, u, lab = define_shared_bounds_labels('_exp', SCENARIOS['_exp'][1], G,
                                            synthetic_two_groups, max_ic=200.0)
    assert len(l) == len(u) == len(lab) == 7
    assert lab[0] == 'r_sensitive' and lab[1] == 'r_resistant'
    assert lab[-1] == 'std'
    assert (l[0], u[0]) == (0.0, 1.0)          # bound of r of _exp


def test_exp_s4_global(synthetic_two_groups):
    l, u, lab = define_shared_bounds_labels('_exp', SCENARIOS['_exp'][4], G,
                                            synthetic_two_groups, max_ic=200.0)
    assert len(l) == 6                          # 1 + 2 + 2 + 1
    assert lab[0] == 'r_global'


def test_log_s1_full_group_specific(synthetic_two_groups):
    l, u, lab = define_shared_bounds_labels('_log', SCENARIOS['_log'][1], G,
                                            synthetic_two_groups, max_ic=200.0)
    assert len(l) == 9                          # 4 + 4 + 1
    assert lab[:4] == ['r_sensitive', 'r_resistant', 'cc_sensitive', 'cc_resistant']


def test_log_s2_cc_global(synthetic_two_groups):
    l, u, lab = define_shared_bounds_labels('_log', SCENARIOS['_log'][2], G,
                                            synthetic_two_groups, max_ic=200.0)
    assert len(l) == 8                          # 3 + 4 + 1
    assert lab[:3] == ['r_sensitive', 'r_resistant', 'cc_global']


def test_log_s4_all_global(synthetic_two_groups):
    l, u, lab = define_shared_bounds_labels('_log', SCENARIOS['_log'][4], G,
                                            synthetic_two_groups, max_ic=200.0)
    assert len(l) == 7                          # 2 + 4 + 1
    assert lab[:2] == ['r_global', 'cc_global']


def test_log_bounds_inherit_phase1_table(synthetic_two_groups):
    """Bounds of r and cc come from bounds.py (r in [0, 1], cc in [100, 20000])."""
    l, u, lab = define_shared_bounds_labels('_log', SCENARIOS['_log'][4], G,
                                            synthetic_two_groups, max_ic=200.0)
    assert (l[0], u[0]) == (0.0, 1.0)           # r_global
    assert (l[1], u[1]) == (100.0, 20000)       # cc_global


def test_ic_and_std_bounds(synthetic_two_groups):
    l, u, lab = define_shared_bounds_labels('_exp', SCENARIOS['_exp'][4], G,
                                            synthetic_two_groups, max_ic=200.0)
    for i in range(1, 5):                        # 4 initial conditions after r_global
        assert (l[i], u[i]) == (0.0, 400.0)     # [0, 2*max_ic]
        assert lab[i].startswith('ic_')
    assert (l[-1], u[-1]) == (0.1, 1000.0)


def test_ic_labels_tagged_by_group(synthetic_two_groups):
    _l, _u, lab = define_shared_bounds_labels('_exp', SCENARIOS['_exp'][4], G,
                                              synthetic_two_groups, max_ic=200.0)
    ics = [x for x in lab if x.startswith('ic_')]
    assert ics == ['ic_sensitive_1', 'ic_sensitive_2',
                   'ic_resistant_1', 'ic_resistant_2']
