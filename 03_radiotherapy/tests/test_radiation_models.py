"""Tests of the radiotherapy models and of the solvers with a dose schedule."""
import numpy as np
import pytest

import radiation_models as rm

GROWTH = (0.089, 3700.0)   # (r, cc)


def test_all_19_models_present():
    assert len(rm.ALL_MODELS) == 19
    assert rm.WINNER_MODEL == '_ExponentialDecayDelay'
    for m in rm.ALL_MODELS:
        assert ('model' + m) in vars(rm)


@pytest.mark.parametrize('model,n_rad', [
    ('_LQ', 2), ('_ExponentialDecay', 2), ('_ExponentialDecayDelay', 3),
    ('_EDEFDelay', 4), ('_EDwithLogFactor', 3), ('_CDSFDelay', 4),
])
def test_n_radiation_params(model, n_rad):
    assert rm.n_radiation_params(model) == n_rad


@pytest.mark.parametrize('model', rm.ALL_MODELS)
def test_every_model_integrates(model, dose_days):
    """All 19 models solve without NaN for reasonable parameters."""
    t = np.array([0., 5., 10., 20., 41.])
    n_rad = rm.n_radiation_params(model)
    # small parameters inside typical bounds
    rad = [0.01, 0.01] + [0.5] * (n_rad - 2)
    sol = rm.solve_model_dispatch(model, t, (*GROWTH, *rad), 100.0, dose_days, 2)
    assert sol.shape == (5, 1)
    assert np.all(np.isfinite(sol))


def test_data_mode_returns_volume_column(dose_days):
    t = np.array([0., 10., 41.])
    for model in ('_ExponentialDecayDelay', '_LQ'):
        sol = rm.solve_model_dispatch(model, t, (*GROWTH, 0.05, 0.05, 1.0)[:2 + rm.n_radiation_params(model)],
                                      100.0, dose_days, 2)
        assert sol.shape == (3, 1)


def test_smooth_mode_prepends_time(dose_days):
    t = np.array([0., 41.])
    sol = rm.solve_model_dispatch('_ExponentialDecayDelay', t,
                                  (*GROWTH, 0.05, 0.05, 1.0), 100.0, dose_days, 2,
                                  type_sol='smooth')
    assert sol.shape[1] == 2
    assert sol[0, 0] == pytest.approx(0.0)
    assert sol[-1, 0] == pytest.approx(41.0)


def test_dose_schedule_matters(dose_days):
    """Changing the dose days changes the trajectory."""
    t = np.array([0., 20., 41.])
    p = (*GROWTH, 0.03, 0.1, 1.0)
    cedo = rm.solve_model_dispatch('_ExponentialDecayDelay', t, p, 100.0,
                                   np.array([0, 1, 2, 3, 4, 5]), 2)[-1, 0]
    tarde = rm.solve_model_dispatch('_ExponentialDecayDelay', t, p, 100.0,
                                    np.array([15, 16, 17, 18, 19, 20]), 2)[-1, 0]
    assert abs(cedo - tarde) > 1.0


def test_treatment_reduces_volume(dose_days):
    """The final volume with radiation is smaller than without (alpha=0)."""
    t = np.array([0., 20., 41.])
    with_rt = rm.solve_model_dispatch('_ExponentialDecayDelay', t,
                                  (*GROWTH, 0.05, 0.1, 1.0), 100.0, dose_days, 2)[-1, 0]
    without_rt = rm.solve_model_dispatch('_ExponentialDecayDelay', t,
                                  (*GROWTH, 0.0, 0.1, 1.0), 100.0, dose_days, 2)[-1, 0]
    assert with_rt < without_rt


def test_lq_discrete_survival_fraction(dose_days):
    """_LQ applies exp(-a d - b d^2) as a discrete jump: more kill -> smaller V."""
    t = np.array([0., 20., 41.])
    strong = rm.solve_model_dispatch('_LQ', t, (*GROWTH, 0.2, 0.05), 100.0, dose_days, 2)[-1, 0]
    weak = rm.solve_model_dispatch('_LQ', t, (*GROWTH, 0.02, 0.005), 100.0, dose_days, 2)[-1, 0]
    no_kill = rm.solve_model_dispatch('_LQ', t, (*GROWTH, 0.0, 0.0), 100.0, dose_days, 2)[-1, 0]
    assert strong < weak < no_kill


def test_lq_no_dose_days_is_pure_growth(dose_days):
    """Without dose days, _LQ is logistic growth."""
    t = np.array([0., 20., 41.])
    no_dose = rm.solve_model_dispatch('_LQ', t, (*GROWTH, 0.2, 0.05), 100.0,
                                       np.array([], dtype=int), 2)[-1, 0]
    # without kill the tumor grows
    assert no_dose > 100.0
