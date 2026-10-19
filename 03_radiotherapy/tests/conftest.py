"""Pytest fixtures of the radiotherapy stage."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.abspath(os.path.dirname(__file__))
_PIPELINE = os.path.abspath(os.path.join(_HERE, '..'))
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)

import _path_setup  # noqa: F401


@pytest.fixture
def dose_days():
    return np.array([0, 1, 2, 3, 4, 5])


@pytest.fixture
def synthetic_rad_data(dose_days):
    """2 groups x 2 mice with dose days."""
    t = np.array([0., 3., 6., 10., 15., 20., 26., 32., 41.])

    def mouse(name, T0, mult):
        V = T0 * (1 + 0.05 * t) * mult          # synthetic increasing curve
        return {'name': name, 'data': np.column_stack((t, V)),
                'treatment': 'radiation', 'treatment_days': dose_days,
                'phenotype': 'synthetic', 'cohort': 'X'}

    return {
        'radiation_sensitive': [mouse('c0_m1', 100., 1.0), mouse('c0_m2', 90., 1.2)],
        'radiation_resistant': [mouse('c1_m1', 110., 1.5), mouse('c1_m2', 105., 1.3)],
    }


@pytest.fixture
def growth_fixed():
    return {
        'radiation_sensitive': dict(r=0.089, cc=3700.0),
        'radiation_resistant': dict(r=0.115, cc=3700.0),
    }
