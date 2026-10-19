"""Pytest fixtures. Puts this stage and 01_control_growth/ (through _path_setup) on sys.path."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.abspath(os.path.dirname(__file__))
_PIPELINE = os.path.abspath(os.path.join(_HERE, '..'))
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)

import _path_setup  # noqa: F401  (adds 01_control_growth/)


@pytest.fixture
def synthetic_two_groups():
    """Synthetic full_data: control_sensitive grows as exp(0.1 t) and control_resistant as exp(0.2 t),
    with 2 mice per group."""
    t = np.array([0, 2, 4, 6, 8, 10], dtype=float)

    def make_mouse(name, r, T0):
        return {'name': name, 'data': np.column_stack((t, T0 * np.exp(r * t))),
                'treatment': 'control',
                'treatment_days': np.array([0, 1, 2, 3, 4, 5]),
                'phenotype': 'synthetic', 'cohort': 'X'}

    return {
        'control_sensitive': [make_mouse('c0_m1', 0.1, 100.0),
                              make_mouse('c0_m2', 0.1,  50.0)],
        'control_resistant': [make_mouse('c1_m1', 0.2,  80.0),
                              make_mouse('c1_m2', 0.2, 120.0)],
    }
