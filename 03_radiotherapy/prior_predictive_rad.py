"""Prior predictive check of the radiotherapy models.

Parameter vectors are drawn from the uniform prior only (without the data), the ODE is solved for each
(inherited growth and effect at the dose days), and the envelope of the curves is drawn with the data
on top. The coverage of the observations under the prior is reported.

Reading:
  - low coverage of the curve band -> the prior does not generate trajectories like the data (wrong
    bounds): the calibration cannot succeed.
  - very wide envelope -> weakly informative prior (not invalid).

    python prior_predictive_rad.py --models _ExponentialDecayDelay --n 300
"""
import _path_setup  # noqa: F401

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from postprocess import configure_plot_settings

from bounds_rad import define_bounds_labels, find_max_initial_condition
from data_loader_rad import load_radiation_data, parse_dose_days_env
from growth_params import load_growth_params
from radiation_models import ALL_MODELS, DOSE_DEFAULT
from postprocess_rad import solve_for_mouse


def sample_prior(l_bound, u_bound, n_draws, rng=None):
    rng = rng or np.random.default_rng()
    l, u = np.asarray(l_bound, float), np.asarray(u_bound, float)
    return rng.uniform(l, u, size=(n_draws, len(l)))


def prior_curves(model_extension, thetas, growth, group, mouse_data, idx, n_ic,
                 dose=DOSE_DEFAULT, type_sol='smooth'):
    """(t_grid, curves (n, T), stds (n,)); parameter vectors whose ODE fails are skipped."""
    curves, stds, t_grid = [], [], None
    for theta in thetas:
        try:
            sol = solve_for_mouse(model_extension, theta, growth, group,
                                  mouse_data, idx, n_ic, type_sol=type_sol,
                                  dose=dose)
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


def prior_coverage(model_extension, thetas, growth, group, mouse_data, idx, n_ic,
                   dose=DOSE_DEFAULT, alpha=0.05, seed=0):
    """(frac_pred, k_pred, frac_curve, k_curve, n_total) under the prior, or None."""
    _t, f, sig = prior_curves(model_extension, thetas, growth, group, mouse_data,
                              idx, n_ic, dose, type_sol='data')
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


def plot_prior_predictive(model_extension, full_data, group, growth, n_draws=300,
                          dose=DOSE_DEFAULT, out_dir='./Output_Calibration',
                          seed=0, alpha=0.05, save=True, max_ic=None):
    """Prior predictive envelope and coverage per mouse. Returns the coverage rows."""
    mice = full_data[group]
    n_ic = len(mice)
    if n_ic == 0:
        return []
    max_ic = max_ic if max_ic is not None else find_max_initial_condition(full_data)
    l, u, _labels = define_bounds_labels(model_extension, n_ic, max_ic,
                                         bool_multiple_mice=True, group=group)
    thetas = sample_prior(l, u, n_draws, np.random.default_rng(seed))

    formatter = configure_plot_settings('14')
    nCols = min(3, n_ic)
    nRows = int(np.ceil(n_ic / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(nCols * 7, nRows * 4.6),
                             squeeze=False)
    axes = axes.ravel()

    rows = []
    for i, m in enumerate(mice):
        ax = axes[i]
        t_grid, f, sig = prior_curves(model_extension, thetas, growth, group, m,
                                      i, n_ic, dose)
        if f is None:
            ax.set_title(f'{m["name"]}: ODE failed for all prior draws')
            continue
        lo, med, hi = np.percentile(f, [100 * alpha / 2, 50,
                                        100 * (1 - alpha / 2)], axis=0)
        ax.fill_between(t_grid, lo, hi, color='cornflowerblue', alpha=0.22, lw=0,
                        label=f'prior {100*(1-alpha):.0f}% (curve)')
        ax.plot(t_grid, med, color='cornflowerblue', lw=1.8, label='prior (median)')
        for k in range(min(25, f.shape[0])):
            ax.plot(t_grid, f[k], color='gray', lw=0.5, alpha=0.35)

        # dose days: dashed green lines
        for k, t_day in enumerate(np.atleast_1d(m['treatment_days'])):
            ax.axvline(x=float(t_day), color='green', linestyle='--', linewidth=1,
                       label='dose' if k == 0 else None)

        ax.plot(m['data'][:, 0], m['data'][:, 1], 'o', color='black', ms=7,
                zorder=5, label='data')

        cov = prior_coverage(model_extension, thetas, growth, group, m, i, n_ic,
                             dose, alpha, seed)
        if cov:
            fp, kp, fc, kc, n_tot = cov
            rows.append(dict(group=group, mouse=m['name'], model=model_extension,
                               coverage_predictive=fp, coverage_curve=fc, inside_curve=kc,
                               total=n_tot))
            ax.set_title(f'{m["name"]}  |  curve: {fc:.0%} ({kc}/{n_tot})  '
                         f'|  predictive: {fp:.0%}', fontsize=11)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.yaxis.set_major_formatter(formatter)
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
                         dose=DOSE_DEFAULT, out_dir='./Output_Calibration',
                         print_table=True, save_csv=True):
    import pandas as pd
    groups = groups or [g for g in full_data if full_data[g]]
    rows = []
    for model in models:
        for g in groups:
            rows += plot_prior_predictive(model, full_data, g, growth,
                                            n_draws=n_draws, dose=dose,
                                            out_dir=out_dir)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if print_table:
        print('\n=== PRIOR PREDICTIVE CHECK (coverage of the data under the prior) ===')
        print(f'  {"model":30s} {"curve":>7s} {"predictive":>10s}   assessment')
        for model, sub in df.groupby('model', sort=False):
            mc, mp = sub['coverage_curve'].mean(), sub['coverage_predictive'].mean()
            flag = ('prior does not generate curves like the data' if mc < 0.3 else
                    'prior covers the data partially' if mc < 0.7 else
                    'prior supports the data')
            print(f'  {model:30s} {mc:>6.0%} {mp:>10.0%}   {flag}')
    if save_csv:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'prior_predictive_coverage.csv')
        df.to_csv(p, index=False, float_format='%.4f')
        if print_table:
            print(f'\nCSV: {p}')
    return df


def _cli():
    ap = argparse.ArgumentParser(description='Prior predictive check (radiotherapy).')
    ap.add_argument('--models', default=','.join(ALL_MODELS))
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--dose', type=float, default=DOSE_DEFAULT)
    ap.add_argument('--dose-days', default=os.environ.get('CALIB_DOSE_DAYS', ''))
    ap.add_argument('--out', default='./Output_Calibration')
    args = ap.parse_args()
    dd = parse_dose_days_env(args.dose_days)
    fd = load_radiation_data(dose_days=dd)
    growth = load_growth_params(list(fd))
    run_prior_predictive(args.models.split(','), fd, growth, n_draws=args.n,
                         dose=args.dose, out_dir=args.out)


if __name__ == '__main__':
    _cli()
