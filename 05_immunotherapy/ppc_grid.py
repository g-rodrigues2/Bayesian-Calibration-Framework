"""Prior predictive check of the twelve immunotherapy-effect models of the thesis (Figure 4.x).

    CALIB_GROWTH_SOURCE=fixed python ppc_grid.py [--draws 2000] [--seed 0]

For each model, parameter vectors are drawn from the uniform prior (bounds_imm.py), the tumor
trajectory of each mouse is simulated with the inherited growth parameters and observation noise
(std from its prior) is added. The 5%-95% band of the simulated observations defines, per mouse,
the coverage (percentage of observed volumes inside the band); each panel reports the median,
interquartile range, mean, minimum and maximum of the coverage across mice. Writes
Output_Calibration/ppc_grid_io_sensitive.png and ppc_coverage_io_sensitive.csv.
"""
import _path_setup  # noqa: F401

import argparse
import os
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import thesis_style as ts
from bounds_imm import define_bounds_labels, extract_treatment_params, find_max_initial_condition
from data_loader_imm import load_immuno_data
from growth_params import load_growth_params
from immuno_models import solve_model

GROUP = 'io_sensitive'
THESIS_MODELS = [
    '_ExponentialDecay', '_CumulativeDose', '_AccelED', '_AccelCD', '_MichaelisMenten', '_dePillisRadunskayaLaw',
    '_CDwithLogFactor', '_CDwithExpFactor', '_CDwithSinFactor', '_EDwithLogFactor', '_EDwithExpFactor',
    '_EDwithSinFactor',
]


def simulate(model, mice, growth, max_ic, n_draws, rng):
    """Per mouse: time grid (union of observed times) and simulated observations (draws x times)."""
    l_bound, u_bound, _ = define_bounds_labels(model, len(mice), max_ic, bool_multiple_mice=True)
    thetas = rng.uniform(l_bound, u_bound, size=(n_draws, len(l_bound)))
    times = np.unique(np.concatenate([m['data'][:, 0] for m in mice]))
    r, cc = growth[GROUP]['r'], growth[GROUP]['cc']
    sims = []
    for i, mouse in enumerate(mice):
        out = np.empty((n_draws, len(times)))
        for k, theta in enumerate(thetas):
            params, n_slots = extract_treatment_params(theta, model, len(mice), i)
            pred = solve_model(model, times, (r, cc, *params), float(theta[n_slots + i]), mouse['treatment_days'], 1.0)[:, 0]
            out[k] = np.clip(pred + rng.normal(0.0, theta[-1], pred.shape), 0.0, None)
        sims.append(out)
    return times, sims


def panel(ax, times, sims, mice, coverage, n_draws, letter):
    """One model, in the layout of the thesis figure."""
    for sim in sims:
        low, up = np.percentile(sim, [5, 95], axis=0)
        ax.fill_between(times, low, up, alpha=0.14, color='cornflowerblue')
    for sim in sims:
        ax.plot(times, sim.mean(axis=0), linewidth=1.0, alpha=0.45, color='black')
    for m in mice:
        ax.scatter(m['data'][:, 0], m['data'][:, 1], s=15, color='red', alpha=0.7)
    ax.set_xlabel('Time (days)', fontsize=20)
    ax.set_ylabel(r'Tumor Volume ($\times 10^3$ mm$^3$)', fontsize=20)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y / 1e3:g}'))
    ax.yaxis.offsetText.set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.text(0.03, 0.97, f'{letter})', transform=ax.transAxes, fontsize=20, va='top', ha='left')
    c = np.asarray(coverage)
    text = (f'Coverage: {np.median(c):.1f}% (IQR: {np.percentile(c, 25):.1f}-{np.percentile(c, 75):.1f}%)\n'
            f'mean={c.mean():.1f}%  min={c.min():.1f}%  max={c.max():.1f}%\n'
            f'mice={len(mice)}  times={len(times)}  draws={n_draws}')
    ax.text(0.1, 0.97, text, transform=ax.transAxes, fontsize=20, fontfamily='Times New Roman', va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='gray'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='./Output_Calibration')
    args = ap.parse_args()

    full = load_immuno_data()
    mice = full[GROUP]
    # growth parameters of the thesis (Section 4.2), independently of CALIB_GROWTH_SOURCE
    growth = load_growth_params([GROUP], source='fixed')
    max_ic = find_max_initial_condition(full)
    rng = np.random.default_rng(args.seed)

    ts.default_font(14)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(4 * 11, 3 * 5))
    rows = []
    for k, (ax, model) in enumerate(zip(axes.ravel(), THESIS_MODELS)):
        times, sims = simulate(model, mice, growth, max_ic, args.draws, rng)
        coverage = []
        for mouse, sim in zip(mice, sims):
            lo, hi = np.percentile(sim, [5, 95], axis=0)
            idx = np.searchsorted(times, mouse['data'][:, 0])
            obs = mouse['data'][:, 1]
            coverage.append(100.0 * np.mean((obs >= lo[idx]) & (obs <= hi[idx])))
            rows.append(dict(model=model, mouse=mouse['name'], coverage_pct=coverage[-1]))
        panel(ax, times, sims, mice, coverage, args.draws, string.ascii_uppercase[k])
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.98, wspace=0.15, hspace=0.1)
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, f'ppc_grid_{GROUP}.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, f'ppc_coverage_{GROUP}.csv'), index=False, float_format='%.2f')
    print(table.groupby('model', sort=False)['coverage_pct'].agg(['mean', 'median', 'min', 'max']).round(1).to_string())


if __name__ == '__main__':
    main()
