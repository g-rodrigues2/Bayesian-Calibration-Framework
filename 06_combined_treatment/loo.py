"""Mouse-level leave-one-out comparison of the formulations with and without SUV (Section 4.7).

    python loo.py train          # calibrate every training fold (one mouse excluded)
    python loo.py strict         # strict out-of-sample prediction and held-out log-likelihood table
    python loo.py conditional    # per-mouse recalibration of initial volume and std (figures only)

Formulations: _SUVLinear (a, b estimated in each fold), _RadImmunoFixed (d fixed at the
immunotherapy estimate; only training initial volumes and std estimated) and _RadImmunoFitD
(no SUV, d estimated in each fold, so one parameter against the two of _SUVLinear). Strict evaluation uses the
maximum-likelihood estimate of each training fold, the first observed volume of the held-out mouse
as initial condition, and scores the remaining observations with the fold std. The conditional
analysis keeps the fold quantities fixed and samples the held-out initial volume and std.
Outputs: Output_Calibration/loo/.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _path_setup  # noqa: F401
from metrics import ccc, pcc

from bounds_ri import STD_BOUNDS, define_bounds_labels
from data_loader_ri import load_data, max_initial_volume
import rad_immuno_fast
from inference_ri import SOLVER, gaussian_loglik, log_probability, split_theta
from mcmc_ri import config_from_env, run_chain
from postprocess_ri import plot_mouse_panel, posterior_draws, trajectories
from rad_immuno_models import solve_mouse
from thesis_params import upstream_params

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = 'rad+io_sensitive'
MODELS = ['_SUVLinear', '_RadImmunoFixed', '_RadImmunoFitD']
OUT = os.path.join(HERE, 'Output_Calibration', 'loo')
REINIT = os.path.join(HERE, 'reinitiate_files')


def fold_stem(model, mouse):
    return f'{GROUP}_EXCL-{mouse}{model}'


def log_probability_conditional(theta, l_bound, u_bound, model, fold_params, mouse, upstream):
    """theta = [ic, std] of the held-out mouse; fold model parameters fixed."""
    if any(not (lo < v < hi) for lo, v, hi in zip(l_bound, theta, u_bound)):
        return -np.inf
    if SOLVER == 'compiled':
        return rad_immuno_fast.log_likelihood(model, fold_params, [theta[0]], theta[1], [mouse], upstream)
    pred = solve_mouse(model, fold_params, theta[0], mouse, upstream)
    return gaussian_loglik(mouse['data'][:, 1], pred, theta[1]) if np.all(np.isfinite(pred)) else -np.inf


def train(mice, upstream, cfg, models=MODELS, selected=None):
    for model in models:
        for mouse in mice:
            if selected and mouse['name'] not in selected:
                continue
            training = [m for m in mice if m['name'] != mouse['name']]
            l_bound, u_bound, labels = define_bounds_labels(model, len(training), max_initial_volume(mice), 'loo')
            print(f'Training {fold_stem(model, mouse["name"])}')
            run_chain(fold_stem(model, mouse['name']), l_bound, u_bound, labels, log_probability,
                      (model, training, upstream), cfg, OUT, REINIT)


def fold_estimate(model, mouse):
    theta = np.load(os.path.join(OUT, f'estimate_{fold_stem(model, mouse)}.npz'))['pars']
    params, _, std = split_theta(theta, model, 0)
    return params, std


def strict(mice, upstream):
    rows = []
    for model in MODELS:
        for mouse in mice:
            params, std = fold_estimate(model, mouse['name'])
            obs = mouse['data'][:, 1]
            pred = solve_mouse(model, params, obs[0], mouse, upstream)
            rows.append(dict(model=model, mouse=mouse['name'], SUV=mouse['SUV'], std_fold=std,
                             heldout_ll=gaussian_loglik(obs[1:], pred[1:], std),
                             rmse=float(np.sqrt(np.mean((pred[1:] - obs[1:]) ** 2))),
                             ccc=ccc(obs[1:], pred[1:]), pcc=pcc(obs[1:], pred[1:])))
    per_mouse = pd.DataFrame(rows)
    per_mouse.to_csv(os.path.join(OUT, 'strict_loo_per_mouse.csv'), index=False, float_format='%.6f')
    wide = per_mouse.pivot(index='mouse', columns='model', values='heldout_ll')
    summary = per_mouse.groupby('model')['heldout_ll'].agg(total='sum', mean='mean').reindex(MODELS)
    best = wide.idxmax(axis=1)
    summary['mice_best'] = [int((best == m).sum()) for m in MODELS]
    reference = '_RadImmunoFixed'
    summary['mice_better_than_fixed'] = [int((wide[m] > wide[reference]).sum()) if m != reference else 0
                                         for m in MODELS]
    summary.to_csv(os.path.join(OUT, 'strict_loo_summary.csv'), float_format='%.4f')
    print(summary.to_string())


def conditional(mice, upstream, cfg, models=MODELS, selected=None):
    max_ic = max_initial_volume(mice)
    l_bound, u_bound = [0.0, STD_BOUNDS['loo'][0]], [2.0 * max_ic, STD_BOUNDS['loo'][1]]
    if 'CALIB_COND_CHAIN_SIZE' in os.environ:
        cfg = dict(cfg, chain_size=int(os.environ['CALIB_COND_CHAIN_SIZE']))
    results = {}
    for model in models:
        for mouse in mice:
            if selected and mouse['name'] not in selected:
                continue
            params, _ = fold_estimate(model, mouse['name'])
            stem = f'{GROUP}_PRED-{mouse["name"]}{model}'
            run_chain(stem, l_bound, u_bound, ['ic', 'std'], log_probability_conditional,
                      (model, params, mouse, upstream), cfg, OUT, REINIT)
            if os.environ.get('CALIB_BACKEND_ONLY', '0') == '1':
                continue
            draws = posterior_draws(os.path.join(OUT, f'chain_{stem}.gz'))
            draws = np.hstack([np.tile(params, (len(draws), 1)), draws]) if params else draws
            tr = trajectories(draws, model, [mouse], upstream)[0]
            results[(model, mouse['name'])] = (tr, np.median([ccc(mouse['data'][:, 1], p) for p in tr['at_obs']]))
    if len(results) == len(MODELS) * len(mice):
        plot_conditional(mice, results)


def plot_conditional(mice, results):
    """Best- and worst-predicted mice (posterior-median CCC of the formulation without SUV), shown for
    both formulations: A/C without SUV, B/D SUV-based."""
    by_name = {m['name']: m for m in mice}
    ranked = sorted((c, name) for (model, name), (_, c) in results.items() if model == '_RadImmunoFixed')
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, (_, name) in enumerate([ranked[-1], ranked[0]]):
        for col, (model, title) in enumerate([('_RadImmunoFixed', 'without SUV'), ('_SUVLinear', 'SUV-based')]):
            plot_mouse_panel(axes[row, col], by_name[name], results[(model, name)][0],
                             f'{"ABCD"[2 * row + col]}   {title}   ', row == 0 and col == 0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'conditional_loo_best_worst.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['train', 'strict', 'conditional'])
    ap.add_argument('--models', default=','.join(MODELS), help='comma-separated formulations')
    ap.add_argument('--mice', default=None, help='comma-separated held-out mice (default: all)')
    args = ap.parse_args()
    models = args.models.split(',')
    selected = args.mice.split(',') if args.mice else None
    os.makedirs(OUT, exist_ok=True)
    mice = load_data()[GROUP]
    upstream = upstream_params('sensitive')
    {'train': lambda: train(mice, upstream, config_from_env(), models, selected),
     'strict': lambda: strict(mice, upstream),
     'conditional': lambda: conditional(mice, upstream, config_from_env(), models, selected)}[args.step]()


if __name__ == '__main__':
    main()
