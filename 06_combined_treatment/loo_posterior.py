"""Posterior predictive leave-one-out log score and per-mouse predictive diagnostics.

    python loo_posterior.py --draws 6000                     # the five formulations of the LOO table
    python loo_posterior.py --models _SUVLinear,_RadImmunoFitD --draws 500

loo.py scores each held-out mouse with the maximum-likelihood estimate of its training fold, which
is a plug-in prediction. This script integrates the same prediction over the posterior of the fold,

    ell_m = sum_j log[ (1/Q) sum_q N( y_mj | T(t_mj; theta^(q)), sigma^(q) ) ],

with theta^(q) and sigma^(q) the draws of the training fold of mouse m, so that the score carries
the uncertainty of the fold instead of one point estimate. The held-out observations never enter the
fold and the initial condition is the first observed volume of the held-out mouse, as in the strict
analysis, so the score remains out of sample. Alongside it the script reports the paired difference
against the refitted comparator M0 (_RadImmunoFitD, d re-estimated in each fold), the count of mice
favored against M0 and against the inherited lysis rate, the error of the posterior mean prediction,
and the empirical coverage and the width of the 95% posterior predictive interval, which is the
mixture over draws and not a band at a single parameter value.

The chains of every fold must have been explored under the inherited vector of thesis_params.py: the
chains of _RadImmunoFitD and _SUVExp written before that vector was re-estimated were retrained
(loo.py train) and the superseded ones kept in Output_Calibration/loo/OLD_pre_retrain_2026-11-10.
Outputs: Output_Calibration/loo/posterior_loo_per_mouse.csv and posterior_loo_summary.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import _path_setup  # noqa: F401
from postprocess import _mixture_quantile

from data_loader_ri import load_data
from inference_ri import split_theta
from loo import GROUP, OUT, fold_setup, fold_stem
from postprocess_ri import posterior_draws
from rad_immuno_models import solve_mouse
from thesis_params import upstream_params

MODELS = ['_SUVLinear', '_SUVExp', '_VolExp', '_RadImmunoFixed', '_RadImmunoFitD']   # LOO table order
REFERENCE = '_RadImmunoFitD'      # M0, the refitted comparator of the LOO table
INHERITED = '_RadImmunoFixed'
N_DRAWS = 2000


def fold_predictions(model, mice, mouse, upstream, n_draws):
    """Held-out trajectories and residual standard deviations over the posterior of the training fold."""
    training, up = fold_setup(mice, model, mouse, upstream)
    draws = posterior_draws(os.path.join(OUT, f'chain_{fold_stem(model, mouse["name"])}.gz'), n_draws)
    obs = mouse['data'][:, 1]
    pred, std = [], []
    for theta in draws:
        params, _, sigma = split_theta(theta, model, len(training))
        pred.append(solve_mouse(model, params, obs[0], mouse, up))
        std.append(sigma)
    return np.array(pred), np.array(std)[:, None]


def score_mouse(model, mice, mouse, upstream, n_draws):
    pred, std = fold_predictions(model, mice, mouse, upstream, n_draws)
    obs = mouse['data'][1:, 1]
    pred = pred[:, 1:]
    logp = -0.5 * ((pred - obs) ** 2 / std ** 2 + np.log(2 * np.pi * std ** 2))
    lo = _mixture_quantile(pred, std, 0.025)
    hi = _mixture_quantile(pred, std, 0.975)
    mean_pred = pred.mean(axis=0)
    return dict(model=model, mouse=mouse['name'], SUV=mouse['SUV'], n_scored=len(obs),
                ell=float(np.sum(logsumexp(logp, axis=0) - np.log(len(pred)))),
                mae=float(np.mean(np.abs(mean_pred - obs))),
                rmse=float(np.sqrt(np.mean((mean_pred - obs) ** 2))),
                coverage=float(np.mean((obs >= lo) & (obs <= hi))),
                width=float(np.mean(hi - lo)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=','.join(MODELS), help='comma-separated formulations')
    ap.add_argument('--draws', type=int, default=N_DRAWS, help='posterior draws per fold')
    args = ap.parse_args()
    models = args.models.split(',')
    mice = load_data()[GROUP]
    upstream = upstream_params('sensitive')

    rows = []
    for model in models:
        for mouse in mice:
            print(f'Scoring {fold_stem(model, mouse["name"])}', flush=True)
            rows.append(score_mouse(model, mice, mouse, upstream, args.draws))
    per_mouse = pd.DataFrame(rows)

    ell = per_mouse.pivot(index='mouse', columns='model', values='ell')
    if REFERENCE in ell:
        per_mouse['delta'] = [r.ell - ell.loc[r.mouse, REFERENCE] for r in per_mouse.itertuples()]
    per_mouse.to_csv(os.path.join(OUT, 'posterior_loo_per_mouse.csv'), index=False, float_format='%.6f')

    summary = per_mouse.groupby('model').agg(total_ell=('ell', 'sum'), mean_ell=('ell', 'mean'),
                                             mean_mae=('mae', 'mean'), mean_rmse=('rmse', 'mean'),
                                             mean_coverage=('coverage', 'mean'),
                                             mean_width=('width', 'mean')).reindex(models)
    for column, ref in (('mice_above_M0', REFERENCE), ('mice_above_inherited', INHERITED)):
        if ref in ell:
            summary[column] = [int((ell[m] > ell[ref]).sum()) if m != ref else pd.NA for m in models]
    summary.to_csv(os.path.join(OUT, 'posterior_loo_summary.csv'), float_format='%.4f')
    print(summary.to_string())
    print()
    for model in [m for m in models if m != REFERENCE and REFERENCE in ell]:
        delta = ell[model] - ell[REFERENCE]
        print(f'delta_m of {model} against M0: favors it in {int((delta > 0).sum())} of {len(delta)} '
              f'mice, median {np.median(delta):.2f}, sum {delta.sum():.2f}, '
              f'range {delta.min():.2f} to {delta.max():.2f}')


if __name__ == '__main__':
    main()
