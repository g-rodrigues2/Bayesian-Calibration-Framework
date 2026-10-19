"""Thesis figures of the combined radiotherapy + immunotherapy cohort (Sections 4.5-4.7).

    python thesis_figures.py [--figures fig12,fig13,fig14,loo] [--out-dir Output_Calibration/thesis_figures]

fig12  prediction with fixed parameters: selected mice and scatter plot (global_fixedpars_PostPredBand),
       every mouse (global_fixedpars_PostPredBand_ALL); chain Output_Calibration/prediction
fig13  SUV-based lysis (linear): global_SUV_based_PostPredBand and _ALL; chain Output_Calibration/suv
fig14  posterior histograms of a and b (fig14a, fig14b)
loo    leave-one-out: best and worst mouse of each formulation and scatter plots (SUV_contribution),
       every mouse (LOO_held_out_SUV_based_ALL, fig_LOO_semSUV_ALL); Output_Calibration/loo

Solid lines are the trajectories of the maximum-likelihood samples (for leave-one-out, the training-fold
estimate and the held-out initial volume); light-blue areas are 95% credible bands and peach areas 95%
posterior predictive intervals (Gaussian observation noise with the sampled std, clipped at zero), both
from 200 posterior draws at the observation times. CCC and PCC are posterior medians and 95% credible
intervals. The random draws follow the original scripts (seed 42), so the figures reproduce their values.
Selected mice: prediction and leave-one-out as in the original scripts (CCC of the maximum-likelihood
trajectory; leave-one-out only mice followed for more than 50 days); SUV-based model by posterior-median CCC.
"""
import argparse
import os
import random
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import _path_setup  # noqa: F401
import thesis_style as ts
from metrics import ccc, pcc

from data_loader_ri import load_data
from inference_ri import split_theta
from rad_immuno_models import solve_mouse
from thesis_params import upstream_params

GROUP = 'rad+io_sensitive'
SEED = 42


def summary(values):
    v = np.asarray(values, float)
    return tuple(np.percentile(v, [50, 2.5, 97.5]))


def treatments(mouse):
    return [(mouse['rad_days'], 'green'), (mouse['io_days'], 'red')]


def bands(latent, predictive):
    return tuple(np.percentile(latent, [97.5, 2.5], axis=0)), tuple(np.percentile(predictive, [97.5, 2.5], axis=0))


# ----------------------------------------------------------------------------- global calibrations
def global_analysis(model, chain, mice, up, draws):
    """Per-mouse and cohort CCC/PCC over `draws` posterior samples, and bands from 200 samples."""
    rng = np.random.default_rng(SEED)
    thetas = chain[rng.choice(len(chain), size=min(draws, len(chain)), replace=False)]
    obs_all = np.concatenate([m['data'][:, 1] for m in mice])
    per_mouse = [[] for _ in mice]
    cohort = []
    for theta in thetas:
        params, ics, _ = split_theta(theta, model, len(mice))
        preds = [solve_mouse(model, params, ics[i], m, up) for i, m in enumerate(mice)]
        for i, m in enumerate(mice):
            per_mouse[i].append((np.clip(ccc(preds[i], m['data'][:, 1]), -1, 1), np.clip(pcc(preds[i], m['data'][:, 1]), -1, 1)))
        cohort.append((np.clip(ccc(np.concatenate(preds), obs_all), -1, 1), np.clip(pcc(np.concatenate(preds), obs_all), -1, 1)))
    mouse_metrics = [(summary(np.array(v)[:, 0]), summary(np.array(v)[:, 1])) for v in per_mouse]
    cohort_metrics = (summary(np.array(cohort)[:, 0]), summary(np.array(cohort)[:, 1]))

    random.seed(SEED)
    np.random.seed(SEED)
    selected = chain[random.sample(range(len(chain)), min(200, len(chain)))]
    mouse_bands = []
    for i, m in enumerate(mice):
        latent, predictive = [], []
        for theta in selected:
            params, ics, std = split_theta(theta, model, len(mice))
            T = solve_mouse(model, params, ics[i], m, up)
            latent.append(T)
            predictive.append(np.clip(T + np.random.normal(0.0, max(std, 0.0), size=T.shape), 0.0, None))
        mouse_bands.append(bands(np.array(latent), np.array(predictive)))
    return mouse_metrics, cohort_metrics, mouse_bands


def global_figures(model, stem, mice, up, draws, out_dir, line_color, selection):
    base = 'prediction' if model == '_RadImmunoFixed' else 'suv'
    chain = np.loadtxt(os.path.join('Output_Calibration', base, f'chain_{GROUP}{model}.gz'), ndmin=2)
    best = np.load(os.path.join('Output_Calibration', base, f'estimate_{GROUP}{model}.npz'))['pars']
    metrics_m, metrics_c, mouse_bands = global_analysis(model, chain, mice, up, draws)
    params, ics, _ = split_theta(best, model, len(mice))
    best_pred = [solve_mouse(model, params, ics[i], m, up) for i, m in enumerate(mice)]
    best_curve = [solve_mouse(model, params, ics[i], m, up, smooth=True) for i, m in enumerate(mice)]
    tmax = max(m['data'][:, 0].max() for m in mice)
    for m, (c, p) in zip(mice, metrics_m):
        print(f'{stem} {m["name"]}: CCC {c[0]:.3f} [{c[1]:.3f}; {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}; {p[2]:.3f}]')
    c, p = metrics_c
    print(f'{stem} cohort: CCC {c[0]:.3f} [{c[1]:.3f}; {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}; {p[2]:.3f}]')

    if selection == 'ml':
        scores = [(i, ccc(best_pred[i], m['data'][:, 1])) for i, m in enumerate(mice) if m['data'][:, 0].max() > 5]
        order = [i for i, _ in sorted(scores, key=lambda x: x[1])]
    else:
        order = list(np.argsort([mm[0][0] for mm in metrics_m]))
    chosen = [order[-1], order[len(order) // 2], order[0]]

    ts.times_font(18)
    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    gs.update(hspace=0.55)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.subplots_adjust(top=0.95)
    pos = axes[3].get_position()
    extra = pos.height * 0.4
    axes[3].set_position([pos.x0, pos.y0 - extra / 2, pos.width, pos.height + extra])
    for k, (ax, i) in enumerate(zip(axes[:3], chosen)):
        m = mice[i]
        ts.fit_panel(ax, m['data'], best_curve[i], 'Prediction', 'selected', band=mouse_bands[i][0],
                     predictive=mouse_bands[i][1], treatments=treatments(m), xmax=tmax, line_color=line_color,
                     legend=(k == 0))
        ts.metrics_text(ax, 'selected', string.ascii_uppercase[k], *metrics_m[i])
        ts.volume_yaxis(ax, 2000)
    ts.scatter_panel(axes[3], [m['data'][:, 1] for m in mice], best_pred, 'D', *metrics_c, style='rtio')
    fig.savefig(os.path.join(out_dir, f'{stem}.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    ts.times_font(18)
    fig, axes = plt.subplots(5, 3, figsize=(33, 25))
    axes = axes.ravel()
    for i, m in enumerate(mice):
        ts.fit_panel(axes[i], m['data'], best_curve[i], 'Prediction', 'all_mice_io', band=mouse_bands[i][0],
                     predictive=mouse_bands[i][1], treatments=treatments(m), xmax=tmax, line_color=line_color,
                     legend=(i == 0))
        ts.metrics_text(axes[i], 'all_mice_io', string.ascii_uppercase[i], *metrics_m[i])
        ts.volume_yaxis(axes[i], 2000)
    for ax in axes[len(mice):]:
        fig.delaxes(ax)
    fig.savefig(os.path.join(out_dir, f'{stem}_ALL.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------------------- leave-one-out
def loo_inputs(model, mice):
    out = {}
    for m in mice:
        name = m['name']
        read = lambda kind: np.loadtxt(os.path.join('Output_Calibration', 'loo', f'chain_{GROUP}_{kind}-{name}{model}.gz'), ndmin=2)
        est = lambda kind: np.load(os.path.join('Output_Calibration', 'loo', f'estimate_{GROUP}_{kind}-{name}{model}.npz'))['pars']
        out[name] = dict(fold=read('EXCL'), fold_best=est('EXCL'), heldout=read('PRED'), heldout_best=est('PRED'))
    return out


def fold_params(model, fold_theta):
    return (float(fold_theta[0]), float(fold_theta[1])) if model == '_SUVLinear' else ()


def loo_analysis(model, mice, up, inputs, draws):
    suv = model == '_SUVLinear'
    random.seed(SEED)
    np.random.seed(SEED)
    metrics_m = []
    for m in mice:
        d = inputs[m['name']]
        selected = d['heldout'][random.sample(range(len(d['heldout'])), min(draws, len(d['heldout'])))]
        values = []
        for theta_ic in selected:
            params = fold_params(model, d['fold'][np.random.randint(len(d['fold']))]) if suv else ()
            T = solve_mouse(model, params, theta_ic[0], m, up)
            values.append((ccc(T, m['data'][:, 1]), pcc(T, m['data'][:, 1])))
        values = np.array(values)
        metrics_m.append((summary(values[:, 0]), summary(values[:, 1])))

    rng = np.random.default_rng(SEED)
    obs_all = np.concatenate([m['data'][:, 1] for m in mice])
    cohort = []
    for _ in range(draws):
        preds = []
        for m in mice:
            d = inputs[m['name']]
            ic = d['heldout'][rng.integers(len(d['heldout'])), 0]
            params = fold_params(model, d['fold'][rng.integers(len(d['fold']))]) if suv else ()
            preds.append(solve_mouse(model, params, ic, m, up))
        cohort.append((ccc(np.concatenate(preds), obs_all), pcc(np.concatenate(preds), obs_all)))
    cohort = np.array(cohort)
    metrics_c = (summary(cohort[:, 0]), summary(cohort[:, 1]))

    random.seed(SEED)
    np.random.seed(SEED)
    mouse_bands = []
    for m in mice:
        d = inputs[m['name']]
        selected = d['heldout'][random.sample(range(len(d['heldout'])), min(200, len(d['heldout'])))]
        latent, predictive = [], []
        for theta_ic in selected:
            if suv:
                theta_fold = d['fold'][np.random.randint(len(d['fold']))]
                params, sigma = fold_params(model, theta_fold), float(theta_fold[-1])
            else:
                params, sigma = (), float(theta_ic[-1])
            T = solve_mouse(model, params, theta_ic[0], m, up)
            latent.append(T)
            predictive.append(np.clip(T + np.random.normal(0.0, max(sigma, 0.0), size=T.shape), 0.0, None))
        mouse_bands.append(bands(np.array(latent), np.array(predictive)))

    best_pred, best_curve = [], []
    for m in mice:
        d = inputs[m['name']]
        params = fold_params(model, d['fold_best'])
        best_pred.append(solve_mouse(model, params, d['heldout_best'][0], m, up))
        best_curve.append(solve_mouse(model, params, d['heldout_best'][0], m, up, smooth=True))
    return metrics_m, metrics_c, mouse_bands, best_pred, best_curve


def loo_figures(mice, up, draws, out_dir):
    tmax = max(m['data'][:, 0].max() for m in mice)
    results = {}
    for model in ('_RadImmunoFixed', '_SUVLinear'):
        results[model] = loo_analysis(model, mice, up, loo_inputs(model, mice), draws)
        metrics_m, metrics_c = results[model][:2]
        for m, (c, p) in zip(mice, metrics_m):
            print(f'LOO{model} {m["name"]}: CCC {c[0]:.3f} [{c[1]:.3f}; {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}; {p[2]:.3f}]')
        c, p = metrics_c
        print(f'LOO{model} cohort: CCC {c[0]:.3f} [{c[1]:.3f}; {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}; {p[2]:.3f}]')

    ts.times_font(18)
    fig = plt.figure(figsize=(24, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    gs.update(hspace=0.25)
    columns = [[fig.add_subplot(gs[r, c]) for r in range(3)] for c in range(2)]
    fig.subplots_adjust(top=1.0)
    for col in columns:
        pos = col[2].get_position()
        col[2].set_position([pos.x0, pos.y0 - 0.1, pos.width, pos.height * 1.4])
    for c, (model, color, letters) in enumerate([('_RadImmunoFixed', 'black', 'ACE'), ('_SUVLinear', 'blue', 'BDF')]):
        metrics_m, metrics_c, mouse_bands, best_pred, best_curve = results[model]
        scores = [(i, ccc(best_pred[i], m['data'][:, 1])) for i, m in enumerate(mice) if m['data'][:, 0].max() > 50]
        order = [i for i, _ in sorted(scores, key=lambda x: x[1])]
        for r, i in enumerate([order[-1], order[0]]):
            ax, m = columns[c][r], mice[i]
            ts.fit_panel(ax, m['data'], best_curve[i], 'LOO Prediction', 'selected', band=mouse_bands[i][0],
                         predictive=mouse_bands[i][1], treatments=treatments(m), xmax=tmax, line_color=color,
                         legend=(r == 0))
            ts.metrics_text(ax, 'selected', letters[r], *metrics_m[i])
            ts.volume_yaxis(ax, 2000)
            print(f'SUV_contribution {letters[r]}: {m["name"]} ({model})')
        ts.scatter_panel(columns[c][2], [m['data'][:, 1] for m in mice], best_pred, letters[2], *metrics_c, style='rtio')
    fig.savefig(os.path.join(out_dir, 'SUV_contribution.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    for model, color, name in (('_SUVLinear', 'blue', 'LOO_held_out_SUV_based_ALL'),
                               ('_RadImmunoFixed', 'black', 'fig_LOO_semSUV_ALL')):
        metrics_m, _, mouse_bands, _, best_curve = results[model]
        ts.times_font(18)
        fig, axes = plt.subplots(5, 3, figsize=(33, 25))
        axes = axes.ravel()
        for i, m in enumerate(mice):
            ts.fit_panel(axes[i], m['data'], best_curve[i], 'LOO Prediction', 'all_mice_io', band=mouse_bands[i][0],
                         predictive=mouse_bands[i][1], treatments=treatments(m), xmax=tmax, line_color=color,
                         legend=(i == 0))
            ts.metrics_text(axes[i], 'all_mice_io', string.ascii_uppercase[i], *metrics_m[i])
            ts.volume_yaxis(axes[i], 2000)
        for ax in axes[len(mice):]:
            fig.delaxes(ax)
        fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)


PANEL_WIDTH = 2.48      # width of each panel of Table 4.18 on the page, in inches


def fig14(out_dir):
    chain = np.loadtxt(os.path.join('Output_Calibration', 'suv', f'chain_{GROUP}_SUVLinear.gz'), ndmin=2)
    est = np.load(os.path.join('Output_Calibration', 'suv', f'estimate_{GROUP}_SUVLinear.npz'), allow_pickle=True)
    labels = [str(x) for x in est['labels']]
    for name, stem, label in (('a', 'fig14a', r'$a$ [day$^{-1}$]'), ('b', 'fig14b', r'$b$ [day$^{-1}$]')):
        i = labels.index(name)
        ts.posterior_histogram(chain[:, i], est['pars'][i], os.path.join(out_dir, stem + '.png'),
                               PANEL_WIDTH, label)
        q = np.percentile(chain[:, i], [50, 2.5, 97.5])
        print(f'{name}: median {q[0]:.4g} [{q[1]:.4g}, {q[2]:.4g}], ML {est["pars"][i]:.4g}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figures', default='fig12,fig13,fig14,loo')
    ap.add_argument('--draws', type=int, default=1000, help='posterior draws for CCC/PCC (original scripts: 1000)')
    ap.add_argument('--out-dir', default='Output_Calibration/thesis_figures')
    args = ap.parse_args()
    which = args.figures.split(',')
    os.makedirs(args.out_dir, exist_ok=True)
    mice = load_data()[GROUP]
    up = upstream_params('sensitive')
    if 'fig12' in which:
        global_figures('_RadImmunoFixed', 'global_fixedpars_PostPredBand', mice, up, args.draws, args.out_dir,
                       'black', 'ml')
    if 'fig13' in which:
        global_figures('_SUVLinear', 'global_SUV_based_PostPredBand', mice, up, args.draws, args.out_dir,
                       'blue', 'posterior')
    if 'fig14' in which:
        fig14(args.out_dir)
    if 'loo' in which:
        loo_figures(mice, up, args.draws, args.out_dir)
    print('figures written to', args.out_dir)


if __name__ == '__main__':
    main()
