"""Tests of the ODE models and solver dispatch."""
import numpy as np
import pytest

from models import ALL_MODELS, _split_theta, solve_model


def test_model_exp_growth():
    """T(t) = T0 exp(r t); for r=0.1, T0=100, t=10: T(10) ~ 271.83."""
    t_arr = np.array([0.0, 5.0, 10.0])
    sol = solve_model('_exp', t_arr, (0.1,), 100.0)
    assert sol.shape == (3, 1)
    np.testing.assert_allclose(sol[0, 0], 100.0, rtol=1e-3)
    np.testing.assert_allclose(sol[-1, 0], 100.0 * np.exp(1.0), rtol=1e-3)


def test_model_log_saturates_at_cc():
    """The logistic model tends to cc for large t."""
    sol = solve_model('_log', np.linspace(0, 100, 50), (1.0, 1000.0), 10.0)
    assert sol[-1, 0] == pytest.approx(1000.0, rel=1e-3)


def test_model_gomp_saturates_at_b_minus_c():
    """Gompertz dT/dt = r T ln(b/(T+c)) has its equilibrium at T = b - c."""
    sol = solve_model('_gomp', np.linspace(0, 200, 50), (0.5, 2000.0, 100.0), 10.0)
    assert sol[-1, 0] == pytest.approx(1900.0, rel=1e-2)


def test_solve_model_smooth_returns_time_column():
    """type_sol='smooth' returns (t, *states) on a grid with dt=0.1."""
    sol = solve_model('_exp', np.array([0.0, 5.0]), (0.1,), 100.0,
                      type_sol='smooth')
    # smooth grid between 0 and 5 with dt=0.1 -> 51 points
    assert sol.shape == (51, 2)
    np.testing.assert_allclose(sol[0, 0], 0.0)
    np.testing.assert_allclose(sol[-1, 0], 5.0)


def test_all_models_integrate_without_error():
    """The 7 models solve with reasonable parameters."""
    params = {
        '_exp':    (0.1,),
        '_mendel': (0.1, 0.9),
        '_bert':   (0.5, 0.05),
        '_lin':    (0.5, 500.0),
        '_gomp':   (0.1, 3000.0, 100.0),
        '_surf':   (0.5, 500.0),
        '_log':    (0.2, 2000.0),
    }
    assert set(params) == set(ALL_MODELS)
    t = np.linspace(0, 20, 10)
    for ext, p in params.items():
        sol = solve_model(ext, t, p, 100.0)
        assert sol.shape == (10, 1)
        assert np.all(np.isfinite(sol)), f"{ext} produced non-finite values"


def test_split_theta_global_mode():
    """Modo global (ndim = n_mice): ode_args = params; ic_fn(idx) = T0_idx."""
    # _exp: theta = [r, T0_1, T0_2, std]
    theta = np.array([0.3, 50.0, 80.0, 10.0])
    ode_args, ic_fn = _split_theta(theta, '_exp', ndim=2)
    assert ode_args == (0.3,)
    assert ic_fn(0) == 50.0
    assert ic_fn(1) == 80.0


def test_split_theta_mouse_specific_mode():
    """Modo mouse-specific (ndim = 1): theta = [params..., ic, std]."""
    # _log: theta = [r, cc, ic, std]
    theta = np.array([0.3, 2000.0, 50.0, 10.0])
    ode_args, ic_fn = _split_theta(theta, '_log', ndim=1)
    assert ode_args == (0.3, 2000.0)
    assert ic_fn(0) == 50.0


def test_odeint_and_solve_ivp_agree(monkeypatch):
    """Both solvers give the same result for a simple model."""
    import models
    t = np.linspace(0, 5, 10)

    monkeypatch.setattr(models, 'SOLVER', 'odeint')
    sol_odeint = models.solve_model('_exp', t, (0.3,), 50.0)

    monkeypatch.setattr(models, 'SOLVER', 'solve_ivp')
    sol_ivp = models.solve_model('_exp', t, (0.3,), 50.0)

    np.testing.assert_allclose(sol_odeint, sol_ivp, rtol=1e-4)
