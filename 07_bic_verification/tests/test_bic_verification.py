import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entries import all_cells, load_estimate  # noqa: E402
from likelihoods import problem, safe  # noqa: E402
from verify_bic import bic, maximize  # noqa: E402


def _cell(cell_id):
    return next(c for c in all_cells() if c['id'] == cell_id)


def test_cell_registry():
    cells = all_cells()
    assert len(cells) == len({c['id'] for c in cells})
    tables = {c['table'] for c in cells}
    assert tables == {'S1.2', '4', 'S2.2', 'S2.3', '5', 'S3.2', 'SUV'}


def test_stored_estimate_reproduces_thesis_bic():
    """Control, mouse C0 M1, exponential model: BIC 173.53 in Table S1.2."""
    cell = _cell('S1.2__control_sensitive__c0_m1___exp')
    pars, stored_ll = load_estimate(cell)
    loglik, lb, ub, _, n = problem(cell)
    ll = safe(loglik)(pars)
    assert abs(ll - stored_ll) < 1e-6
    assert round(bic(ll, len(lb), n), 2) == 173.53


def test_linear_quadratic_model_reproduces_original_calibration():
    cell = _cell('S2.2__radiation_sensitive__c0_m16___LQ')
    pars, stored_ll = load_estimate(cell)
    loglik = problem(cell)[0]
    assert abs(safe(loglik)(pars) - stored_ll) < 1e-3


def test_maximize_never_decreases_the_start():
    rng = np.random.default_rng(0)
    target = np.array([0.3, -0.2])

    def f(x):
        return -float(np.sum((x - target) ** 2))
    start = rng.uniform(-1, 1, size=2)
    theta, ll, winner, _ = maximize(f, [-1, -1], [1, 1], [start], maxfev=500, rounds=3)
    assert ll >= f(start)
    assert np.allclose(theta, target, atol=1e-4)
