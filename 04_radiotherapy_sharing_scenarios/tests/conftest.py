"""Pytest fixtures of the radiotherapy sharing scenarios."""
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
def rad_two_groups():
    t = np.array([0., 3., 6., 10., 20., 41.])
    days = np.array([0, 1, 2, 3, 4, 5])

    def mouse(name, T0):
        return {'name': name, 'data': np.column_stack((t, T0 * (1 + 0.03 * t))),
                'treatment': 'radiation', 'treatment_days': days,
                'phenotype': 'synthetic', 'cohort': 'X'}

    return {
        'radiation_sensitive': [mouse('c0_m1', 100.), mouse('c0_m2', 90.)],
        'radiation_resistant': [mouse('c1_m1', 110.), mouse('c1_m2', 105.)],
    }


@pytest.fixture
def growth_fixed():
    return {'radiation_sensitive': dict(r=0.089, cc=3700.0),
            'radiation_resistant': dict(r=0.115, cc=3700.0)}
