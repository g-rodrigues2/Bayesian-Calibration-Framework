"""Prior predictive check of the immunotherapy models (curve and predictive coverage per mouse).

With three mice the prior carries much weight: if the uniform box cannot generate curves similar to the
data, the calibration cannot succeed. Procedure:
  1. draw N parameter vectors from the uniform prior (l_bound, u_bound);
  2. solve the ODE for each (inherited growth, effect at the administration days);
  3. draw the envelope (percentiles) of the curves with the data on top;
  4. report the coverage: fraction of observations inside the prior band.

The figure of the thesis (twelve models, coverage of the 5-95% band of simulated observations) is
produced by ppc_grid.py.

    python prior_predictive.py --models _dePillisRadunskayaLaw --n 300
"""
import _path_setup  # noqa: F401

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from postprocess import configure_plot_settings

from bounds_imm import define_bounds_labels, extract_treatment_params, \
    find_max_initial_condition
from data_loader_imm import load_immuno_data, parse_dose_days_env
from growth_params import load_growth_params
from immuno_models import DOSE_DEFAULT, solve_model


def sample_prior(l_bound, u_bound, n_draws, rng=None):
    """n_draws samples of the uniform prior on the box (one row per parameter vector)."""
    rng = rng or np.random.default_rng()
    l, u = np.asarray(l_bound, float), np.asarray(u_bound, float)
    return rng.uniform(l, u, size=(n_draws, len(l)))


def prior_predictive_curves(model_extension, thetas, growth, group, mouse_data,
                            idx, n_ic, ms_params=(), dose=DOSE_DEFAULT,
                            type_sol='smooth'):
    """(t_grid, curves (n, T), stds (n,)); parameter vectors whose ODE fails are skipped."""
    r, cc = growth[group]['r'], growth[group]['cc']
    curves, stds, t_grid = [], [], None
    for theta in thetas:
        params, n_slots = extract_treatment_params(
            theta, model_extension, n_ic, idx, ms_params if n_ic > 1 else ())
        T0 = float(theta[n_slots:n_slots + n_ic][idx])
        try:
            sol = solve_model(model_extension, mouse_data['data'][:, 0],
                              (r, cc, *params), T0, mouse_data['treatment_days'],
                              dose, type_sol=type_sol)
        except Exception:
            continue
        y = sol[:, 1] if type_sol == 'smooth' else sol[:, 0]
        if not np.all(np.isfinite(y)):
            continue
        if type_sol == 'smooth':
            t_grid = sol[:, 0]
        curves.append(y)
        stds.append(abs(float(theta[-1])))
    if not curves:
        return None, None, None
    return t_grid, np.vstack(curves), np.asarray(stds)


def prior_predictive_coverage(model_extension, thetas, growth, group, mouse_data,
                              idx, n_ic, ms_params=(), dose=DOSE_DEFAULT,
                              alpha=0.05, seed=0):
    """Two coverages of the observations under the prior:

      predictive -- band of y = f + noise(std). The prior of `std` is wide, so this band tends to
                    contain everything.
      curve      -- band of the curves f(t) only; it shows whether the model can produce trajectories
                    like the data under the prior.

    Returns (frac_pred, k_pred, frac_curve, k_curve, n_total), or None.
    """
    _t, f, sig = prior_predictive_curves(
        model_extension, thetas, growth, group, mouse_data, idx, n_ic,
        ms_params, dose, type_sol='data')
    if f is None:
        return None
    obs = mouse_data['data'][:, 1]
    qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]

    rng = np.random.default_rng(seed)
    draws = f + sig[:, None] * rng.normal(size=f.shape)
    plo, phi = np.percentile(draws, qs, axis=0)
    inside_p = (obs >= plo) & (obs <= phi)

    clo, chi = np.percentile(f, qs, axis=0)
    inside_c = (obs >= clo) & (obs <= chi)

    return (float(inside_p.mean()), int(inside_p.sum()),
            float(inside_c.mean()), int(inside_c.sum()), int(len(obs)))


def plot_prior_predictive(model_extension, full_data, group, growth,
                          n_draws=300, ms_params=(), dose=DOSE_DEFAULT,
                          out_dir='./Output_Calibration', seed=0, alpha=0.05,
                          save=True, max_ic=None):
    """Prior predictive envelope and coverage per mouse. Returns the coverage rows (one per mouse)."""
    mice = full_data[group]
    n_ic = len(mice)
    max_ic = max_ic if max_ic is not None else find_max_initial_condition(full_data)
    l, u, _labels = define_bounds_labels(model_extension, n_ic, max_ic,
                                         bool_multiple_mice=True,
                                         ms_params=ms_params)
    rng = np.random.default_rng(seed)
    thetas = sample_prior(l, u, n_draws, rng)

    formatter = configure_plot_settings('14')
    nCols = min(3, n_ic)
    nRows = int(np.ceil(n_ic / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(nCols * 7, nRows * 4.6),
                             squeeze=False)
    axes = axes.ravel()

    rows = []
    for i, m in enumerate(mice):
        ax = axes[i]
        t_grid, f, sig = prior_predictive_curves(
            model_extension, thetas, growth, group, m, i, n_ic, ms_params, dose)
        if f is None:
            ax.set_title(f'{m["name"]}: ODE failed for all prior draws')
            continue

        # prior envelope of the curves (percentiles)
        lo, med, hi = np.percentile(f, [100 * alpha / 2, 50,
                                        100 * (1 - alpha / 2)], axis=0)
        ax.fill_between(t_grid, lo, hi, color='darkorange', alpha=0.22, lw=0,
                        label=f'prior {100*(1-alpha):.0f}% (curve)')
        ax.plot(t_grid, med, color='darkorange', lw=1.8, label='prior (median)')
        # some individual trajectories
        for k in range(min(25, f.shape[0])):
            ax.plot(t_grid, f[k], color='gray', lw=0.5, alpha=0.35)

        # administration days: dashed red lines
        for k, t_day in enumerate(np.atleast_1d(m['treatment_days'])):
            ax.axvline(x=float(t_day), color='red', linestyle='--', linewidth=1,
                       label='administration' if k == 0 else None)

        ax.plot(m['data'][:, 0], m['data'][:, 1], 'o', color='black', ms=7,
                zorder=5, label='data')

        cov = prior_predictive_coverage(model_extension, thetas, growth, group,
                                        m, i, n_ic, ms_params, dose, alpha, seed)
        if cov:
            fp, kp, fc, kc, n_tot = cov
            rows.append(dict(group=group, mouse=m['name'],
                               model=model_extension, coverage_predictive=fp,
                               coverage_curve=fc, inside_curve=kc, total=n_tot))
            ax.set_title(f'{m["name"]}  |  curve: {fc:.0%} ({kc}/{n_tot})  '
                         f'|  predictive: {fp:.0%}', fontsize=11)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.yaxis.set_major_formatter(formatter)
        # keep the data visible even when prior trajectories diverge
        ymax = max(np.percentile(hi, 95), m['data'][:, 1].max() * 1.6)
        ax.set_ylim(0, ymax)
        if i == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    for j in range(n_ic, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f'Prior predictive check — {model_extension} ({group}), '
                 f'{n_draws} prior draws', fontsize=13)
    fig.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f'prior_predictive_{group}{model_extension}')
        fig.savefig(base + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(base + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return rows


def run_prior_predictive(models, full_data, growth, groups=None, n_draws=300,
                         ms_params=(), dose=DOSE_DEFAULT,
                         out_dir='./Output_Calibration', print_table=True,
                         save_csv=True):
    """Runs the check for each (model, group). Returns the coverage DataFrame."""
    import pandas as pd
    groups = groups or list(full_data)
    rows = []
    for model in models:
        for g in groups:
            rows += plot_prior_predictive(model, full_data, g, growth,
                                            n_draws=n_draws, ms_params=ms_params,
                                            dose=dose, out_dir=out_dir)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    if print_table:
        print('\n=== PRIOR PREDICTIVE CHECK (coverage of the data under the prior) ===')
        print(f'  {"model":32s} {"curve":>7s} {"predictive":>10s}   assessment')
        for model, sub in df.sort_values('coverage_curve', ascending=False).groupby(
                'model', sort=False):
            mc = sub['coverage_curve'].mean()
            mp = sub['coverage_predictive'].mean()
            flag = ''
            if mc < 0.3:
                flag = 'prior does not generate curves like the data'
            elif mc < 0.7:
                flag = 'prior covers the data partially'
            else:
                flag = 'prior supports the data'
            if mp > 0.99:
                flag += ' (wide predictive band: wide prior of std)'
            print(f'  {model:32s} {mc:>6.0%} {mp:>10.0%}   {flag}')
        print('\n  curve = envelope of f(t) under the prior')
        print('  predictive = curve + noise(std); saturates with a wide prior of std')

    if save_csv:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'prior_predictive_coverage.csv')
        df.to_csv(p, index=False, float_format='%.4f')
        if print_table:
            print(f'\nCSV: {p}')
    return df


def _cli():
    from bounds_imm import CALIBRATABLE_MODELS
    ap = argparse.ArgumentParser(description='Prior predictive check (immunotherapy).')
    ap.add_argument('--models', default=','.join(CALIBRATABLE_MODELS))
    ap.add_argument('--n', type=int, default=300, help='prior draws')
    ap.add_argument('--ms-params', default='',
                    help='parameters calibrated per mouse (e.g. d)')
    ap.add_argument('--dose', type=float, default=DOSE_DEFAULT)
    ap.add_argument('--dose-days', default=os.environ.get('CALIB_DOSE_DAYS', ''))
    ap.add_argument('--out', default='./Output_Calibration')
    args = ap.parse_args()

    dd = parse_dose_days_env(args.dose_days)
    fd = load_immuno_data(dose_days=dd)
    growth = load_growth_params(list(fd))
    ms = tuple(x for x in args.ms_params.split(',') if x.strip())
    run_prior_predictive(args.models.split(','), fd, growth, n_draws=args.n,
                         ms_params=ms, dose=args.dose, out_dir=args.out)


if __name__ == '__main__':
    _cli()
