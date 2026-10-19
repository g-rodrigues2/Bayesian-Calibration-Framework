"""Profile log-likelihood of the treatment parameters of an immunotherapy model (global calibration).

For each treatment parameter, the parameter is fixed on a grid and the remaining parameters (other treatment
parameters, initial conditions and std) are optimized inside their prior bounds (bounded Powell). The
optimizations start from the posterior samples whose value of the profiled parameter is closest to the grid
value and from the maximum-likelihood sample, which avoids the local optima of a single start along the
curved ridges of saturating models.

A profile that stays within 1.92 log-likelihood units of its maximum (the 95% threshold of a chi-squared with
one degree of freedom) over a wide part of the range indicates practical non-identifiability: the data do not
constrain that parameter, and its posterior is shaped by the prior bounds rather than by the likelihood.

    CALIB_GROWTH_SOURCE=fixed python profile_likelihood.py --model _dePillisRadunskayaLaw \\
        --chain Output_Calibration/multi_chain_io_sensitive_dePillisRadunskayaLaw.gz --extend d=50 --lower lam=0.3

--extend widens the grid of a parameter beyond its prior upper bound (the other parameters keep their
prior bounds), to check whether the likelihood keeps increasing outside the prior.
"""
import argparse
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import _path_setup  # noqa: F401
from growth_params import load_growth_params
from bounds_imm import define_bounds_labels, find_max_initial_condition, treatment_param_names
from data_loader_imm import load_immuno_data
from inference_imm import log_likelihood

GROUP = 'io_sensitive'
_STATE = {}


def _init(model):
    data = load_immuno_data()
    _STATE.update(model=model, data=data, growth=load_growth_params([GROUP]))


def _profile_point(task):
    """Maximum log-likelihood with parameter `index` fixed at `value`."""
    index, value, starts, l_bound, u_bound = task
    model, data, growth = _STATE['model'], _STATE['data'], _STATE['growth']
    free = [i for i in range(len(l_bound)) if i != index]
    eps = 1e-9 * (u_bound - l_bound)
    bounds = list(zip(l_bound[free] + eps[free], u_bound[free] - eps[free]))
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def neg(x):
        theta = np.empty(len(l_bound))
        theta[free] = x
        theta[index] = value
        v = log_likelihood(theta, model, data, GROUP, growth)
        return -v if np.isfinite(v) else 1e12

    best = None
    for start in starts:
        res = minimize(neg, np.clip(start[free], lo, hi), method='Powell', bounds=bounds,
                       options=dict(maxiter=4000, xtol=1e-5, ftol=1e-8))
        if best is None or res.fun < best.fun:
            best = res
    theta = np.empty(len(l_bound))
    theta[free] = best.x
    theta[index] = value
    return index, value, -best.fun, theta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='_dePillisRadunskayaLaw')
    ap.add_argument('--chain', help='posterior chain (.gz) used for the starting points (not needed with --plot-only)')
    ap.add_argument('--points', type=int, default=15)
    ap.add_argument('--starts', type=int, default=4, help='posterior samples used as starts per grid value')
    ap.add_argument('--extend', nargs='*', default=[], help='PARAM=UPPER grid limits beyond the prior')
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--passes', type=int, default=2, help='continuation passes from neighboring optima')
    ap.add_argument('--lower', nargs='*', default=[], help='PARAM=LOWER grid limits (default: prior lower bound)')
    ap.add_argument('--out', default='./Output_Calibration')
    ap.add_argument('--plot-only', action='store_true', help='redraw the figure from the saved table')
    args = ap.parse_args()

    if args.plot_only:
        data = load_immuno_data()
        l_bound, u_bound, labels = define_bounds_labels(
            args.model, group_size=len(data[GROUP]), max_ic=find_max_initial_condition(data), bool_multiple_mice=True)
        stem = f'{args.out}/profile_likelihood_{GROUP}{args.model}'
        plot_profile(pd.read_csv(stem + '.csv'), treatment_param_names(args.model), labels, np.asarray(l_bound, float),
                     np.asarray(u_bound, float), stem + '.png')
        print(f'[profile] {stem}.png')
        return
    if not args.chain:
        ap.error('--chain is required unless --plot-only is given')

    _init(args.model)
    data = _STATE['data']
    l_bound, u_bound, labels = define_bounds_labels(
        args.model, group_size=len(data[GROUP]), max_ic=find_max_initial_condition(data),
        bool_multiple_mice=True)
    l_bound, u_bound = np.asarray(l_bound, float), np.asarray(u_bound, float)
    names = treatment_param_names(args.model)
    extend = {k: float(v) for k, v in (kv.split('=') for kv in args.extend)}
    grid_lower = {k: float(v) for k, v in (kv.split('=') for kv in args.lower)}
    chain = np.loadtxt(args.chain, ndmin=2)
    growth = _STATE['growth']
    ll_chain = np.array([log_likelihood(t, args.model, data, GROUP, growth)
                         for t in chain[np.linspace(0, len(chain) - 1, 2000).astype(int)]])
    ml_sample = chain[np.linspace(0, len(chain) - 1, 2000).astype(int)][int(np.argmax(ll_chain))]

    tasks = []
    for i, name in enumerate(names):
        upper = extend.get(name, u_bound[i])
        grid = np.geomspace(grid_lower.get(name, l_bound[i]), upper, args.points)
        box_u = u_bound.copy()
        box_u[i] = max(u_bound[i], upper)
        order = np.argsort(np.abs(np.log(chain[:, i]) - np.log(grid[:, None])), axis=1)[:, :args.starts]
        for g, value in enumerate(grid):
            starts = [chain[j] for j in order[g]] + [ml_sample]
            tasks.append((i, value, starts, l_bound, box_u))

    with Pool(args.cores, initializer=_init, initargs=(args.model,)) as pool:
        results = pool.map(_profile_point, tasks)
        # continuation passes: each grid value is re-optimized from the optima of its neighbors
        for _ in range(args.passes):
            refined = []
            for k, (i, value, ll, theta) in enumerate(results):
                neighbors = [results[j][3] for j in (k - 1, k + 1)
                             if 0 <= j < len(results) and results[j][0] == i]
                refined.append((i, value, [theta] + neighbors, tasks[k][3], tasks[k][4]))
            new = pool.map(_profile_point, refined)
            results = [n if n[2] > r[2] else r for n, r in zip(new, results)]

    rows = [dict(parameter=names[i], value=v, log_likelihood=ll,
                 **{f'at_{n}': th[j] for j, n in enumerate(labels)}) for i, v, ll, th in results]
    table = pd.DataFrame(rows)
    inside = [(r.parameter, r.value) for r in table.itertuples()
              if r.value <= u_bound[labels.index(r.parameter)]]
    ll_max_inside = table[[(p, v) in inside for p, v in zip(table.parameter, table.value)]].log_likelihood.max()
    table['delta_log_likelihood'] = table['log_likelihood'] - ll_max_inside
    os.makedirs(args.out, exist_ok=True)
    stem = f'{args.out}/profile_likelihood_{GROUP}{args.model}'
    table.to_csv(stem + '.csv', index=False, float_format='%.6g')
    print(table[['parameter', 'value', 'log_likelihood', 'delta_log_likelihood']].to_string(index=False))

    plot_profile(table, names, labels, l_bound, u_bound, stem + '.png')
    print(f'[profile] {stem}.csv, {stem}.png')


def plot_profile(table, names, labels, l_bound, u_bound, path, floor=-12.0):
    """One panel per parameter; values below `floor` are drawn at the floor (open markers)."""
    import thesis_style as ts
    ts.times_font(12)
    symbols = {'d': r'$d$ [day$^{-1}$]', 's': r'$s$ [(mm$^3$)$^{-\lambda}$]', 'lam': r'$\lambda$'}
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.8))
    for k, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        part = table[table['parameter'] == name].sort_values('value')
        i = labels.index(name)
        y = part['delta_log_likelihood'].to_numpy()
        x = part['value'].to_numpy()
        ax.axvspan(l_bound[i], u_bound[i], color='gray', alpha=0.10, lw=0, label='prior range')
        ax.plot(x, np.maximum(y, floor), '-', color='#2c7fb8', lw=1.5)
        ax.plot(x[y >= floor], y[y >= floor], 'o', color='#2c7fb8', ms=4, label='profile')
        ax.plot(x[y < floor], np.full((y < floor).sum(), floor), 'v', mfc='white', color='#2c7fb8', ms=5,
                label=f'below {floor:g}')
        ax.axhline(-1.92, color='crimson', ls='--', lw=1, label='95% threshold')
        if x.max() / x.min() > 20:
            ax.set_xscale('log')
            ax.set_xlim(x.min() / 1.3, x.max() * 1.3)
        else:
            pad = 0.05 * (x.max() - x.min())
            ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(floor - 0.5, 1.0)
        ax.set_xlabel(symbols.get(name, name), fontsize=12)
        ax.text(0.03, 0.93, f'{chr(65 + k)})', transform=ax.transAxes, fontsize=13)
    np.atleast_1d(axes)[0].set_ylabel('profile log-likelihood (relative)', fontsize=12)
    np.atleast_1d(axes)[1 if len(names) > 1 else 0].legend(fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
