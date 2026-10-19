"""Shared pytest fixtures. Puts the stage folder on sys.path so that its modules can be imported."""
import os
import sys

import numpy as np
import pytest

# tests/.. is the stage folder that holds the modules
_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)


@pytest.fixture
def synthetic_mouse_data():
    """One synthetic mouse with known exponential growth: T(t) = 100 exp(0.1 t), 6 points in [0, 10]."""
    t = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    T = 100.0 * np.exp(0.1 * t)
    return np.column_stack((t, T))


def _mouse(name, data, cohort):
    return {'name': name, 'data': data, 'treatment': 'control',
            'treatment_days': np.array([0, 1, 2, 3, 4, 5]),
            'phenotype': 'synthetic', 'cohort': cohort}


@pytest.fixture
def synthetic_full_data(synthetic_mouse_data):
    """Minimal full_data with the two TNBC control groups: control_sensitive has 2 mice and
    control_resistant has 1.
    """
    return {
        'control_sensitive': [
            _mouse('c0_m1', synthetic_mouse_data, '0'),
            _mouse('c0_m2', synthetic_mouse_data * np.array([1.0, 1.5]), '0'),
        ],
        'control_resistant': [
            _mouse('c1_m1', synthetic_mouse_data * np.array([1.0, 2.0]), '1'),
        ],
    }
