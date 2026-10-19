import os
import sys

import numpy as np
from numba import njit
from scipy.integrate import odeint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compiled_ode  # noqa: E402
import models  # noqa: E402

TIMES = np.array([0, 1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 21, 24, 28, 31, 35, 38, 42, 45, 49, 52, 56.])


@njit
def _delayed_decay(y, t, r, cc, alpha, beta, delay, tau, dose):
    effect = 0.0
    for i in range(tau.size):
        if t >= tau[i] + delay:
            effect += alpha * dose[i] * np.exp(-beta * (t - tau[i] - delay))
    return r * y * (1 - y / cc) - effect * y


@njit
def _indexed_volume(y, t, r, cc, d, tau, dose):
    tumor = max(y[0], 1e-12)
    effect = 0.0
    for i in range(tau.size):
        if t >= tau[i]:
            effect += d / (1.0 + tumor / 500.0)
    return r * tumor * (1 - tumor / cc) - effect * tumor


def _reference(fn, pars, y0):
    return odeint(fn, [y0], TIMES, args=pars, rtol=1e-12, atol=1e-10, mxstep=100000)[:, 0]


def test_growth_laws_match_odeint():
    for model, pars in [('_bert', (0.9, 0.05)), ('_gomp', (0.1, 5000.0, 1.0)), ('_lin', (40.0, 200.0)),
                        ('_mendel', (0.3, 0.8)), ('_surf', (1.0, 20.0))]:
        fn = getattr(models, 'model' + model)
        ref = _reference(fn, pars, 120.0)
        new = compiled_ode.solve(fn, TIMES, pars, 120.0)[:, 0]
        assert np.allclose(new, ref, rtol=1e-7, atol=1e-6), model


def test_effect_switching_on_after_a_delay():
    tau, dose = np.arange(6.0), np.full(6, 2.0)
    for delay in (0.0, 2.5, 4.0, 7.9):
        pars = (0.09, 3700.0, 0.05, 0.3, delay)
        ref = _reference(_delayed_decay, (*pars, tau, dose), 120.0)
        new = compiled_ode.solve(_delayed_decay, TIMES, pars, 120.0, tau, dose, delay)[:, 0]
        assert np.allclose(new, ref, rtol=1e-7, atol=1e-6), delay


def test_right_hand_side_indexing_the_volume():
    tau, dose = np.arange(0.0, 19.0, 3.0), np.ones(7)
    pars = (0.09, 3700.0, 0.2)
    ref = _reference(_indexed_volume, (*pars, tau, dose), 150.0)
    new = compiled_ode.solve(_indexed_volume, TIMES, pars, 150.0, tau, dose)[:, 0]
    assert np.allclose(new, ref, rtol=1e-7, atol=1e-6)


@njit
def _narrow_pulse(y, t, r, cc, alpha, beta, tau, dose):
    effect = 0.0
    for i in range(tau.size):
        if t >= tau[i]:
            effect += alpha * dose[i] * np.exp(-beta * (t - tau[i]) ** 2)
    return r * y * (1 - y / cc) - effect * y


def test_narrow_effect_pulses_are_resolved():
    """With a decay rate of 80/day^2 each administration acts for about 0.2 day; the reference restarts a
    tight DOP853 integration at every administration."""
    from scipy.integrate import solve_ivp
    tau, dose = np.arange(0.0, 19.0, 3.0), np.ones(7)
    pars = (0.09, 3700.0, 0.7, 80.0)
    ref, y, t_prev = [150.0], [150.0], 0.0
    stops = np.unique(np.concatenate((tau[1:], [TIMES[-1]])))
    values = {0.0: 150.0}
    for stop in stops:
        t_eval = TIMES[(TIMES > t_prev) & (TIMES <= stop)]
        sol = solve_ivp(lambda t, v: [_narrow_pulse.py_func(v[0], min(t, stop - 1e-10), *pars, tau, dose)],
                        (t_prev, stop), y, method='DOP853', rtol=1e-12, atol=1e-10, first_step=1e-5,
                        t_eval=np.unique(np.r_[t_eval, stop]))
        values.update(zip(sol.t, sol.y[0]))
        y, t_prev = [sol.y[0][-1]], stop
    ref = np.array([values[t] for t in TIMES])
    new = compiled_ode.solve(_narrow_pulse, TIMES, pars, 150.0, tau, dose)[:, 0]
    assert np.allclose(new, ref, rtol=1e-6)
