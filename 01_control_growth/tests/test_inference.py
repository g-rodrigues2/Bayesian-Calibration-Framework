"""Tests of log_likelihood, log_prior and log_probability in both modes."""
import numpy as np

from inference import log_likelihood, log_prior, log_probability

GROUP = 'control_sensitive'  # 2 mice in the fixture


def test_log_prior_inside_bounds_returns_zero():
    assert log_prior([0.5, 100.0], [0.0, 0.0], [1.0, 200.0]) == 0.0


def test_log_prior_outside_bounds_returns_neg_inf():
    assert log_prior([1.5, 100.0], [0.0, 0.0], [1.0, 200.0]) == -np.inf
    assert log_prior([0.5, -10.0], [0.0, 0.0], [1.0, 200.0]) == -np.inf


def test_log_likelihood_finite_on_good_params(synthetic_full_data):
    """For data T = 100 exp(0.1 t) and theta near the truth, ll is finite."""
    theta = np.array([0.1, 100.0, 100.0, 10.0])  # [r, T0_1, T0_2, std]
    ll = log_likelihood(theta, '_exp', synthetic_full_data, GROUP)
    assert np.isfinite(ll)


def test_log_likelihood_prefers_true_params(synthetic_full_data):
    """r=0.1 (true) must beat r=0.5 (wrong)."""
    good = np.array([0.1, 100.0, 100.0, 10.0])
    bad  = np.array([0.5, 100.0, 100.0, 10.0])
    ll_good = log_likelihood(good, '_exp', synthetic_full_data, GROUP)
    ll_bad  = log_likelihood(bad, '_exp', synthetic_full_data, GROUP)
    assert ll_good > ll_bad


def test_log_likelihood_returns_float_on_bad_params(synthetic_full_data):
    """Pathological parameters must not stop the sampler."""
    # _bert with a dominant b: T becomes negative and T^(2/3) is NaN
    theta = np.array([0.001, 0.5, 100.0, 100.0, 10.0])
    ll = log_likelihood(theta, '_bert', synthetic_full_data, GROUP)
    assert isinstance(ll, float)
    assert not np.isnan(ll)


def test_log_likelihood_handles_solve_model_exception(monkeypatch, synthetic_full_data):
    """If solve_model raises, log_likelihood returns -inf."""
    import inference

    def bad_solve(*a, **kw):
        raise RuntimeError("falha intencional")

    monkeypatch.setattr(inference, 'solve_model', bad_solve)
    theta = np.array([0.1, 100.0, 100.0, 10.0])
    assert log_likelihood(theta, '_exp', synthetic_full_data, GROUP) == -np.inf


def test_log_likelihood_handles_nan_in_solution(monkeypatch, synthetic_full_data):
    """If the solution has NaN, log_likelihood returns -inf."""
    import inference

    def nan_solve(*a, **kw):
        out = np.full((6, 1), 100.0)
        out[2, 0] = np.nan
        return out

    monkeypatch.setattr(inference, 'solve_model', nan_solve)
    theta = np.array([0.1, 100.0, 100.0, 10.0])
    assert log_likelihood(theta, '_exp', synthetic_full_data, GROUP) == -np.inf


# === Modo mouse-specific ===

def test_log_likelihood_mouse_index_uses_single_mouse(synthetic_full_data):
    """With mouse_index, theta = [r, ic, std] (a single initial condition)."""
    theta = np.array([0.1, 100.0, 10.0])
    ll = log_likelihood(theta, '_exp', synthetic_full_data, GROUP,
                        mouse_index=0)
    assert np.isfinite(ll)


def test_mouse_index_selects_the_right_mouse(synthetic_full_data):
    """Mouse 0 has T0=100 and mouse 1 has T0=150; theta with ic=100 must fit mouse 0 better."""
    theta = np.array([0.1, 100.0, 10.0])
    ll0 = log_likelihood(theta, '_exp', synthetic_full_data, GROUP, mouse_index=0)
    ll1 = log_likelihood(theta, '_exp', synthetic_full_data, GROUP, mouse_index=1)
    assert ll0 > ll1


def test_joint_equals_sum_of_singles(synthetic_full_data):
    """With the same growth parameters, std and initial conditions, the joint likelihood (global mode)
    equals the sum of the individual likelihoods (mouse-specific mode).
    """
    r, std = 0.1, 10.0
    ic0, ic1 = 100.0, 150.0

    theta_joint = np.array([r, ic0, ic1, std])
    ll_joint = log_likelihood(theta_joint, '_exp', synthetic_full_data, GROUP)

    ll_sum = sum(
        log_likelihood(np.array([r, ic, std]), '_exp', synthetic_full_data,
                       GROUP, mouse_index=i)
        for i, ic in enumerate([ic0, ic1])
    )

    np.testing.assert_allclose(ll_joint, ll_sum, rtol=1e-12)


# === log_probability ===

def test_log_probability_combines_prior_and_likelihood(synthetic_full_data):
    """log_probability = log_prior + log_likelihood inside the bounds."""
    theta = np.array([0.1, 100.0, 100.0, 10.0])
    l = [0.0, 0.0, 0.0, 0.1]
    u = [1.0, 1000.0, 1000.0, 1000.0]
    lp = log_probability(theta, l, u, '_exp', synthetic_full_data, GROUP)
    ll = log_likelihood(theta, '_exp', synthetic_full_data, GROUP)
    np.testing.assert_allclose(lp, ll)  # prior = 0 inside the box


def test_log_probability_neg_inf_outside_bounds(synthetic_full_data):
    """Outside the bounds -> -inf, without evaluating the likelihood."""
    theta = np.array([2.0, 100.0, 100.0, 10.0])  # r=2 outside [0, 1]
    l = [0.0, 0.0, 0.0, 0.1]
    u = [1.0, 1000.0, 1000.0, 1000.0]
    assert log_probability(theta, l, u, '_exp', synthetic_full_data,
                           GROUP) == -np.inf


def test_log_probability_forwards_mouse_index(synthetic_full_data):
    """log_probability passes mouse_index to the likelihood."""
    theta = np.array([0.1, 100.0, 10.0])
    l = [0.0, 0.0, 0.1]
    u = [1.0, 1000.0, 1000.0]
    lp = log_probability(theta, l, u, '_exp', synthetic_full_data, GROUP,
                         mouse_index=0)
    ll = log_likelihood(theta, '_exp', synthetic_full_data, GROUP, mouse_index=0)
    np.testing.assert_allclose(lp, ll)
