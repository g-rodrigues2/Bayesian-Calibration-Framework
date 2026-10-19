"""Pytest fixtures of the immunotherapy stage."""
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
    """Experimental schedule: 7 administrations every 3 days."""
    return np.array([0, 3, 6, 9, 12, 15, 18])


@pytest.fixture
def synthetic_imm_data(dose_days):
    """One group (io_sensitive) with 3 mice and SUV values."""
    t = np.array([0., 3., 6., 10., 15., 20., 26., 32.])

    def mouse(name, T0, suv):
        return {'name': name, 'data': np.column_stack((t, T0 * (1 + 0.04 * t))),
                'treatment': 'immuno', 'treatment_days': dose_days,
                'SUV': suv, 'phenotype': 'sensitive', 'cohort': '0'}

    return {'io_sensitive': [mouse('c0_m3', 100., 0.73),
                             mouse('c0_m28', 110., 1.16),
                             mouse('c0_m35', 95., 1.20)]}


@pytest.fixture
def growth_fixed():
    return {'io_sensitive': dict(r=0.088, cc=3741.0)}
