"""Posterior trajectories, concordance metrics and figures (Figures 4.12, 4.13 and Appendix)."""
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _path_setup  # noqa: F401
from metrics import ccc, pcc
from postprocess import _mixture_quantile

from inference_ri import split_theta
from rad_immuno_models import solve_mouse

plt.rcParams.update({'font.size': 12})


def posterior_draws(chain_path, n_draws=200, seed=42):
    chain = np.loadtxt(chain_path, ndmin=2)
    idx = np.random.default_rng(seed).choice(len(chain), size=min(n_draws, len(chain)), replace=False)
    return chain[idx]


def trajectories(draws, model, mice, upstream):
    """For each mouse: smooth time grid, curves (draws x grid), stds, and predictions at the
    observed times (draws x n_obs)."""
    out = []
    for i, mouse in enumerate(mice):
        curves, at_obs, stds = [], [], []
        for theta in draws:
            params, ics, std = split_theta(theta, model, len(mice))
            t, T = solve_mouse(model, params, ics[i], mouse, upstream, smooth=True)
            curves.append(T)
            at_obs.append(np.interp(mouse['data'][:, 0], t, T))
            stds.append(std)
        out.append(dict(t=t, curves=np.array(curves), at_obs=np.array(at_obs), std=np.array(stds)))
    return out


def concordance(traj, mice):
    """Per-mouse posterior-median CCC/PCC and cohort CCC/PCC (median and 95% interval over draws)."""
    per_mouse = pd.DataFrame([dict(mouse=m['name'],
                                   ccc=np.median([ccc(m['data'][:, 1], p) for p in tr['at_obs']]),
                                   pcc=np.median([pcc(m['data'][:, 1], p) for p in tr['at_obs']]))
                              for m, tr in zip(mice, traj)])
    obs = np.concatenate([m['data'][:, 1] for m in mice])
    draws = np.hstack([tr['at_obs'] for tr in traj])
    cohort = {name: np.percentile([f(obs, d) for d in draws], [50, 2.5, 97.5]) for name, f in (('ccc', ccc), ('pcc', pcc))}
    return per_mouse, cohort


def plot_mouse_panel(ax, mouse, tr, label, show_legend):
    lo, hi = np.percentile(tr['curves'], [2.5, 97.5], axis=0)
    sigma = tr['std'][:, None]
    ax.fill_between(tr['t'], _mixture_quantile(tr['curves'], sigma, 0.025),
                    _mixture_quantile(tr['curves'], sigma, 0.975), color='#fdd0a2', alpha=0.6,
                    label='95% posterior predictive interval')
    ax.fill_between(tr['t'], lo, hi, color='#6baed6', alpha=0.5, label='95% credible band')
    ax.plot(tr['t'], np.median(tr['curves'], axis=0), color='#08519c', lw=2, label='posterior median')
    ax.plot(mouse['data'][:, 0], mouse['data'][:, 1], 'ko', ms=5, label='data')
    for k, day in enumerate(mouse['rad_days']):
        ax.axvline(day, color='green', ls='--', lw=0.8, label='radiotherapy' if k == 0 else None)
    for k, day in enumerate(mouse['io_days']):
        ax.axvline(day, color='red', ls='--', lw=0.8, label='immunotherapy' if k == 0 else None)
    ccc_m = np.median([ccc(mouse['data'][:, 1], p) for p in tr['at_obs']])
    pcc_m = np.median([pcc(mouse['data'][:, 1], p) for p in tr['at_obs']])
    ax.set_title(f'{label}{mouse["name"]}   CCC {ccc_m:.2f}   PCC {pcc_m:.2f}', fontsize=11)
    ax.set_xlabel('time (days)')
    ax.set_ylabel('tumor volume (mm$^3$)')
    if show_legend:
        ax.legend(fontsize=8, loc='upper left')


def plot_summary(traj, mice, per_mouse, cohort, path):
    """Best, intermediate and worst mouse by posterior-median CCC, and the cohort scatter plot."""
    order = per_mouse.sort_values('ccc', ascending=False).index.to_list()
    chosen = [order[0], order[len(order) // 2], order[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for k, (ax, i) in enumerate(zip(axes.ravel()[:3], chosen)):
        plot_mouse_panel(ax, mice[i], traj[i], f'{string.ascii_uppercase[k]}   ', k == 0)
    ax = axes[1, 1]
    obs = np.concatenate([m['data'][:, 1] for m in mice])
    pred = np.concatenate([np.median(tr['at_obs'], axis=0) for tr in traj])
    ax.scatter(obs, pred, s=18, color='#08519c', alpha=0.7)
    top = 1.05 * max(obs.max(), pred.max())
    ax.plot([0, top], [0, top], 'k--', lw=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel('observed tumor volume (mm$^3$)')
    ax.set_ylabel('predicted tumor volume (mm$^3$)')
    ax.set_title(f'D   CCC {cohort["ccc"][0]:.3f} [{cohort["ccc"][1]:.3f}, {cohort["ccc"][2]:.3f}]   '
                 f'PCC {cohort["pcc"][0]:.3f} [{cohort["pcc"][1]:.3f}, {cohort["pcc"][2]:.3f}]', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_all_mice(traj, mice, path, n_cols=4):
    n_rows = int(np.ceil(len(mice) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for k, (ax, mouse, tr) in enumerate(zip(axes.ravel(), mice, traj)):
        plot_mouse_panel(ax, mouse, tr, f'{string.ascii_uppercase[k]}   ', k == 0)
    for ax in axes.ravel()[len(mice):]:
        ax.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def figures_from_chain(chain_path, model, mice, upstream, out_prefix, n_draws=200):
    traj = trajectories(posterior_draws(chain_path, n_draws), model, mice, upstream)
    per_mouse, cohort = concordance(traj, mice)
    per_mouse.to_csv(f'{out_prefix}_concordance.csv', index=False)
    plot_summary(traj, mice, per_mouse, cohort, f'{out_prefix}_summary.png')
    plot_all_mice(traj, mice, f'{out_prefix}_all_mice.png')
    return per_mouse, cohort
