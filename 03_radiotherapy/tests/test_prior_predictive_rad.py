"""Tests of the radiotherapy prior predictive check."""
import numpy as np

import prior_predictive_rad as pp


def test_sample_prior_within_box():
    l, u = [0.0, 0.0, 0.1], [0.1, 8.0, 1000.0]
    s = pp.sample_prior(l, u, 200, np.random.default_rng(0))
    assert s.shape == (200, 3)
    assert np.all(s >= np.array(l)) and np.all(s <= np.array(u))


def test_prior_curves_and_coverage(synthetic_rad_data, growth_fixed):
    m = '_ExponentialDecayDelay'
    g = 'radiation_resistant'
    n_ic = len(synthetic_rad_data[g])
    # prior draws: [alpha, beta, delay, ic1, ic2, std]
    rng = np.random.default_rng(1)
    thetas = np.column_stack([
        rng.uniform(0, 0.05, 40), rng.uniform(0, 0.1, 40),
        rng.uniform(0, 3, 40), rng.uniform(80, 120, 40),
        rng.uniform(80, 120, 40), rng.uniform(20, 60, 40)])
    t, f, sig = pp.prior_curves(m, thetas, growth_fixed, g,
                                synthetic_rad_data[g][0], 0, n_ic)
    assert f is not None and f.shape[1] == len(t)
    cov = pp.prior_coverage(m, thetas, growth_fixed, g,
                            synthetic_rad_data[g][0], 0, n_ic)
    assert cov is not None
    fp, kp, fc, kc, n = cov
    assert 0.0 <= fc <= 1.0 and 0.0 <= fp <= 1.0
    assert kc <= n


def test_prior_curves_none_when_all_fail(monkeypatch, synthetic_rad_data,
                                         growth_fixed):
    import prior_predictive_rad
    monkeypatch.setattr(prior_predictive_rad, 'solve_for_mouse',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    thetas = np.ones((5, 6))
    t, f, sig = pp.prior_curves('_ExponentialDecayDelay', thetas, growth_fixed,
                                'radiation_resistant',
                                synthetic_rad_data['radiation_resistant'][0], 0, 2)
    assert f is None


def test_plot_and_run_write_files(tmp_path, synthetic_rad_data, growth_fixed):
    df = pp.run_prior_predictive(['_ExponentialDecay'], synthetic_rad_data,
                                 growth_fixed, n_draws=30,
                                 out_dir=str(tmp_path), print_table=False)
    assert not df.empty
    assert set(['group', 'mouse', 'model', 'coverage_curve', 'coverage_predictive']).issubset(df.columns)
    import os
    figs = [f for f in os.listdir(tmp_path) if f.startswith('prior_predictive_')
            and f.endswith('.png')]
    assert len(figs) >= 2   # 2 groups
    assert os.path.exists(tmp_path / 'prior_predictive_coverage.csv')
