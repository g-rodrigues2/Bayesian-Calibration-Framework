"""Post-processing of the radiotherapy calibration: figures and summaries, with the treatment solution
(inherited growth and effect anchored at the dose days).

Produces:
  - time courses per mouse with the 95% credible band, the 95% predictive interval and the dose days
  - observed versus predicted scatter plots (per group and combined) with CCC/PCC and credible intervals
  - posterior histograms of the radiotherapy parameters
  - corner and trace plots (written by global.py)

Reuses the generic helpers of the control stage (scales, axes, scatter panels, HDI).
"""
import _path_setup  # noqa: F401

import os
import string

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr

from metrics import ccc, pcc
from postprocess import (
    configure_plot_settings, finalize_plot, plot_scatter_ax, summarize_posterior,
)
from radiation_models import n_radiation_params, solve_model_dispatch
from inference_rad import DOSE

GROUP_LEFT = 'radiation_sensitive'
GROUP_RIGHT = 'radiation_resistant'


def split_theta_rad(theta, model_extension, n_ic):
    """theta = [radiotherapy parameters..., ic_1..ic_n, std] -> (rad_args, ic_fn)."""
    n_rad = n_radiation_params(model_extension)
    rad = tuple(theta[:n_rad])
    T0s = theta[n_rad:n_rad + n_ic]
    return rad, (lambda i: float(T0s[i]))


def solve_for_mouse(model_extension, theta, growth, group, mouse_data, idx,
                    n_ic, type_sol='data', dose=DOSE):
    rad, ic_fn = split_theta_rad(theta, model_extension, n_ic)
    r, cc = growth[group]['r'], growth[group]['cc']
    return solve_model_dispatch(
        model_extension, mouse_data['data'][:, 0], (r, cc, *rad), ic_fn(idx),
        mouse_data['treatment_days'], dose, type_sol=type_sol)


# === Posterior bands (as in postprocess.posterior_bands) ===

def _mixture_quantile(f, sigma, q, max_iter=60):
    lo = (f - 10.0 * sigma).min(axis=0)
    hi = (f + 10.0 * sigma).max(axis=0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cdf = ndtr((mid[None, :] - f) / sigma).mean(axis=0)
        below = cdf < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def posterior_bands_rad(chain, model_extension, growth, group, mouse_data, idx,
                        n_ic, n_samp=200, seed=42, alpha=0.05, dose=DOSE):
    """{'t', 'median', 'credible': (lo, hi), 'predictive': (lo, hi)}, or None.

    credible   = percentiles of f(t; theta) (parameter uncertainty)
    predictive = quantiles of y ~ N(f, std^2) (adds the observation noise)
    """
    chain = np.asarray(chain)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    n_samp = min(n_samp, chain.shape[0])
    rng = np.random.default_rng(seed)
    sel = rng.choice(chain.shape[0], size=n_samp, replace=False)

    curves, stds, t_grid = [], [], None
    for theta in chain[sel]:
        try:
            sol = solve_for_mouse(model_extension, theta, growth, group,
                                  mouse_data, idx, n_ic, type_sol='smooth',
                                  dose=dose)
        except Exception:
            continue
        if not np.all(np.isfinite(sol[:, 1])):
            continue
        t_grid = sol[:, 0]
        curves.append(sol[:, 1])
        stds.append(abs(float(theta[-1])))
    if not curves:
        return None

    f = np.vstack(curves)
    sigma = np.maximum(np.asarray(stds)[:, None], 1e-12)
    qs = [100 * alpha / 2, 50, 100 * (1 - alpha / 2)]
    c_lo, med, c_hi = np.percentile(f, qs, axis=0)
    return {'t': t_grid, 'median': med, 'credible': (c_lo, c_hi),
            'predictive': (_mixture_quantile(f, sigma, alpha / 2),
                           _mixture_quantile(f, sigma, 1 - alpha / 2))}


# === CCC/PCC with 95% credible intervals ===

def posterior_corr_rad(chain_path, model_extension, growth, full_data, group,
                       mouse_index=None, n_samp=500, seed=42, dose=DOSE):
    if not os.path.exists(chain_path):
        return None
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    mice = (full_data[group] if mouse_index is None
            else [full_data[group][mouse_index]])
    n_ic = len(mice)

    n_samp = min(n_samp, chain.shape[0])
    rng = np.random.default_rng(seed)
    thetas = chain[rng.choice(chain.shape[0], size=n_samp, replace=False)]

    ccs, pcs = [], []
    for theta in thetas:
        try:
            mod, obs = [], []
            for i, m in enumerate(mice):
                sol = solve_for_mouse(model_extension, theta, growth, group, m,
                                      i, n_ic, dose=dose)
                mod.append(sol[:, 0]); obs.append(m['data'][:, 1])
            mod, obs = np.concatenate(mod), np.concatenate(obs)
            ccs.append(float(np.clip(ccc(mod, obs), -1, 1)))
            pcs.append(float(np.clip(pcc(mod, obs), -1, 1)))
        except Exception:
            continue
    if not ccs:
        return None
    cs, ps = summarize_posterior(np.array(ccs)), summarize_posterior(np.array(pcs))
    tag = group if mouse_index is None else f"{group}[{mice[0]['name']}]"
    print(f'\n[Posterior {tag} | {model_extension}]  n={len(ccs)}')
    print(f'PCC: median={ps[0]:.3f}, 95% CrI=[{ps[1]:.3f}, {ps[2]:.3f}]')
    print(f'CCC: median={cs[0]:.3f}, 95% CrI=[{cs[1]:.3f}, {cs[2]:.3f}]')
    return {'ccc_summary': cs, 'pcc_summary': ps,
            'ccc_samples': np.array(ccs), 'pcc_samples': np.array(pcs)}


# === Time courses with bands and dose days ===

def _panel(ax, mouse_data, model_extension, theta, growth, group, idx, n_ic,
           max_t, formatter, band=None, central='median', dose=DOSE):
    data = mouse_data['data']
    ax.set_ylabel('Tumor Volume (mm³)')
    ax.set_xlabel('Time (days)')
    ax.text(0.05, .9, f'{string.ascii_uppercase[idx % 26]}) {mouse_data["name"]}',
            transform=ax.transAxes, fontsize=18)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlim((-1, max_t + 1))

    # Dose days: dashed green vertical lines
    for k, t_day in enumerate(np.atleast_1d(mouse_data['treatment_days'])):
        ax.axvline(x=float(t_day), color='green', linestyle='--', linewidth=1,
                   label='dose' if k == 0 else None)

    if band is not None:
        t = band['t']
        plo, phi = band['predictive']
        ax.fill_between(t, plo, phi, color='steelblue', alpha=0.13, lw=0,
                        label='95% predictive interval')
        clo, chi = band['credible']
        ax.fill_between(t, clo, chi, color='steelblue', alpha=0.35, lw=0,
                        label='95% credible band')
        if central == 'median':
            ax.plot(t, band['median'], color='steelblue', lw=2,
                    label='Model (median)')

    sol = solve_for_mouse(model_extension, theta, growth, group, mouse_data, idx,
                          n_ic, type_sol='smooth', dose=dose)
    ax.plot(sol[:, 0], sol[:, 1], color='black', ls='dashed', lw=2,
            label='Model (max-LL)')
    ax.plot(data[:, 0], data[:, 1], 'o', color='black', ms=9, zorder=5,
            label='Tumor Volume')

    sol_d = solve_for_mouse(model_extension, theta, growth, group, mouse_data,
                            idx, n_ic, dose=dose)[:, 0]
    ax.text(0.5, .05, f'CCC/PCC = {ccc(sol_d, data[:,1]):.2f}/'
                      f'{pcc(sol_d, data[:,1]):.2f}', transform=ax.transAxes)
    if idx == 0:
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
                  bbox_to_anchor=(0.0, 0.86))
    return data[:, 1], sol_d


def plot_maxll_rad(full_data, growth, group, model_extension, out_dir,
                   mode='global', nCols=4, band=True, n_band=200,
                   central='median', dose=DOSE, save=True, figure_name=None):
    """One panel per mouse. mode='global' uses the estimate of the group and 'mouse_specific' the
    estimate of each mouse. Returns (observed, model) for the scatter plot."""
    formatter = configure_plot_settings('14')
    mice = full_data[group]
    max_t = max(max(m['data'][:, 0]) for m in mice)

    entries = []
    if mode == 'global':
        npz = os.path.join(out_dir, f'multi_ll_pars_{group}{model_extension}.npz')
        gz = os.path.join(out_dir, f'multi_chain_{group}{model_extension}.gz')
        if not os.path.exists(npz):
            return None
        theta = np.load(npz, allow_pickle=True)['pars']
        chain = np.loadtxt(gz) if (band and os.path.exists(gz)) else None
        for i, m in enumerate(mice):
            entries.append((i, m, theta, chain, len(mice)))
    else:
        for i, m in enumerate(mice):
            stem = f"{group}_{m['name']}{model_extension}"
            npz = os.path.join(out_dir, f'll_pars_{stem}.npz')
            gz = os.path.join(out_dir, f'chain_{stem}.gz')
            if not os.path.exists(npz):
                continue
            theta = np.load(npz, allow_pickle=True)['pars']
            chain = np.loadtxt(gz) if (band and os.path.exists(gz)) else None
            entries.append((i, m, theta, chain, 1))
    if not entries:
        return None

    n = len(entries)
    nRows = int(np.ceil(n / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(nCols * 11, nRows * 5))
    axes = np.atleast_1d(axes).ravel()

    exp_all, mod_all = [], []
    for panel, (i, m, theta, chain, n_ic) in enumerate(entries):
        idx_band = i if n_ic > 1 else 0
        b = None
        if chain is not None:
            b = posterior_bands_rad(chain, model_extension, growth, group, m,
                                    idx_band, n_ic, n_samp=n_band, dose=dose)
        e, mo = _panel(axes[panel], m, model_extension, theta, growth, group,
                       idx_band if n_ic > 1 else 0, n_ic, max_t, formatter,
                       band=b, central=central, dose=dose)
        exp_all.append(e); mod_all.append(mo)

    figure_name = figure_name or os.path.join(
        out_dir, f'max_ll_{group}{model_extension}')
    finalize_plot(fig, axes, n, nCols, nRows, exp_all, mod_all, save, False,
                  figure_name, formatter)
    return exp_all, mod_all


def scatter_both_rad(full_data, growth, model_extension, out_dir, mode='global',
                     groups=(GROUP_LEFT, GROUP_RIGHT), dose=DOSE, save=True):
    """Side-by-side scatter plots of the two groups, with CCC/PCC and 95% credible intervals."""
    formatter = configure_plot_settings('14')
    panels = []
    for g in groups:
        r = plot_maxll_rad(full_data, growth, g, model_extension, out_dir, mode,
                           band=False, save=False, dose=dose)
        if r is None:
            return
        exp, mod = r
        if mode == 'global':
            gz = os.path.join(out_dir, f'multi_chain_{g}{model_extension}.gz')
            res = posterior_corr_rad(gz, model_extension, growth, full_data, g,
                                     dose=dose)
        else:
            res = None
        panels.append((exp, mod, res))
    plt.close('all')

    fig, axes = plt.subplots(1, 2, figsize=(22, 15 * (1.4 / 3.4)))
    for ax, (exp, mod, res) in zip(axes, panels):
        if res:
            c, p = res['ccc_summary'], res['pcc_summary']
            plot_scatter_ax(ax, exp, mod, formatter, ccc_val=c[0],
                            ccc_ic=[c[1], c[2]], pcc_val=p[0], pcc_ic=[p[1], p[2]])
        else:
            plot_scatter_ax(ax, exp, mod, formatter, None, None, None, None)
    fig.tight_layout()
    if save:
        base = os.path.join(out_dir, f'scatter_both{model_extension}')
        fig.savefig(base + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(base + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
