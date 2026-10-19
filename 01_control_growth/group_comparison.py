"""Comparison of posterior distributions between groups: box plots and Mann-Whitney U test.

For a given model, each parameter gets a panel:
  - model parameters (r, cc, ...) and `std`: one box per group
  - `ic`: one box per mouse, with the groups separated by a gap

A two-sided Mann-Whitney U test between the groups is reported on each panel. It is a non-parametric
test of distribution shift and does not assume normal posteriors.

Interpretation: posterior samples are autocorrelated and their number is set by the MCMC run, so the
p-value is optimistic. It indicates separation of the posteriors, not a formal hypothesis test.

    python group_comparison.py --models _log
    python group_comparison.py --mode mouse_specific --models _log
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from postprocess import is_model_param, summarize_posterior

DEFAULT_GROUPS = ['control_sensitive', 'control_resistant']
DEFAULT_MODELS = ['_exp', '_mendel', '_bert', '_lin', '_gomp', '_surf', '_log']

_CAIXA = dict(showfliers=False, patch_artist=True,
              medianprops=dict(color='red', linewidth=1))
_COR = '#9ecae1'   # sns.color_palette("Blues")[2]


def _paths(results_dir, mode, group, model, mouse=None):
    if mode == 'global':
        d = os.path.join(results_dir, 'Output_Calibration')
        s = f'{group}{model}'
        return (os.path.join(d, f'multi_chain_{s}.gz'),
                os.path.join(d, f'multi_ll_pars_{s}.npz'))
    d = os.path.join(results_dir, 'Output_Calibration_per_mouse')
    s = f'{group}_{mouse}{model}'
    return (os.path.join(d, f'chain_{s}.gz'),
            os.path.join(d, f'll_pars_{s}.npz'))


def _load(chain_path, npz_path):
    """(chain, labels, best_pars), or (None, None, None) if a file is missing."""
    if not (os.path.exists(chain_path) and os.path.exists(npz_path)):
        return None, None, None
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    npz = np.load(npz_path, allow_pickle=True)
    labels = [str(x) for x in npz['labels']] if 'labels' in npz else None
    if labels is None or len(labels) != chain.shape[1]:
        return None, None, None
    return chain, labels, np.asarray(npz['pars'])


def collect_group_posteriors(results_dir, group, model, mode='global',
                             full_data=None):
    """{param: {'samples': array, 'best': float}} of a group.

    Global mode: the single chain of the group; initial conditions are 'ic1'..'icN'.
    Mouse-specific mode: the chains of the mice are concatenated for the model parameters and std;
    each mouse contributes its own 'ic'.
    """
    if mode == 'global':
        chain, labels, best = _load(*_paths(results_dir, mode, group, model))
        if chain is None:
            return {}
        return {lbl: {'samples': chain[:, i], 'best': float(best[i])}
                for i, lbl in enumerate(labels)}

    if full_data is None:
        raise ValueError('mouse_specific mode requires full_data')

    pooled = {}
    for mouse in full_data[group]:
        name = mouse['name']
        chain, labels, best = _load(*_paths(results_dir, mode, group, model,
                                            name))
        if chain is None:
            continue
        for i, lbl in enumerate(labels):
            # the 'ic' of each mouse is a separate entry
            key = f'ic[{name}]' if lbl == 'ic' else lbl
            if key in pooled:
                pooled[key]['samples'] = np.concatenate(
                    [pooled[key]['samples'], chain[:, i]])
            else:
                pooled[key] = {'samples': chain[:, i],
                                    'best': float(best[i])}
    return pooled


def mannwhitney(a, b):
    """(U, p) of the two-sided test; (nan, nan) if a side is empty."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float('nan'), float('nan')
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return float('nan'), 1.0
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    return float(u), float(p)


def _shared_params(by_group):
    """Model parameters (and std) present in all groups, in order."""
    if not by_group:
        return []
    common = set.intersection(*(set(d) for d in by_group.values()))
    model_params = [p for p in common if is_model_param(p)]
    order = list(next(iter(by_group.values())))
    model_params.sort(key=order.index)
    return model_params + (['std'] if 'std' in common else [])


def compare_groups(results_dir, groups, model, mode='global', full_data=None):
    """Rows with the Mann-Whitney U and p of each shared parameter."""
    by_group = {g: collect_group_posteriors(results_dir, g, model, mode,
                                             full_data)
                 for g in groups}
    by_group = {g: d for g, d in by_group.items() if d}
    if len(by_group) < 2:
        return [], by_group

    names = list(by_group)
    rows = []
    for par in _shared_params(by_group):
        a = by_group[names[0]][par]['samples']
        b = by_group[names[1]][par]['samples']
        u, p = mannwhitney(a, b)
        med_a = summarize_posterior(a)
        med_b = summarize_posterior(b)
        rows.append(dict(
            model=model, param=par,
            group_a=names[0], median_a=med_a[0], lo_a=med_a[1], hi_a=med_a[2],
            group_b=names[1], median_b=med_b[0], lo_b=med_b[1], hi_b=med_b[2],
            U=u, p=p, significant=bool(np.isfinite(p) and p < 0.05),
            n_a=a.size, n_b=b.size,
        ))
    return rows, by_group


def plot_group_boxplots(results_dir, groups, model, mode='global',
                        full_data=None, out_dir=None, nCols=3,
                        show=False, save=True):
    """One panel per parameter: boxes of the groups and the Mann-Whitney p-value in the title.

    Initial conditions get their own panel, with one box per mouse and a gap between the groups.
    """
    rows, by_group = compare_groups(results_dir, groups, model, mode, full_data)
    if not rows:
        print(f'[group_comparison] no chains for {model} in {results_dir}')
        return rows

    out_dir = out_dir or os.path.join(results_dir, 'Output_Calibration')
    os.makedirs(out_dir, exist_ok=True)
    names = list(by_group)
    tick_labels = [g.replace('control_', '').capitalize() for g in names]

    pars = [r['param'] for r in rows]
    has_ic = any(any(p.startswith('ic') for p in d) for d in by_group.values())
    panels = pars + (['ic'] if has_ic else [])

    nRows = int(np.ceil(len(panels) / nCols))
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols,
                             figsize=(nCols * 5.5, nRows * 4.5))
    axes = np.atleast_1d(axes).ravel()

    for ax, par in zip(axes, panels):
        if par == 'ic':
            samples, best_values, xlbls = [], [], []
            for gi, g in enumerate(names):
                ics = sorted(p for p in by_group[g] if p.startswith('ic'))
                for p in ics:
                    samples.append(by_group[g][p]['samples'])
                    best_values.append(by_group[g][p]['best'])
                    xlbls.append('')
                if gi == 0 and len(names) > 1:       # gap between the groups
                    samples.append([]); best_values.append(np.nan); xlbls.append('')
            width = 0.8
        else:
            samples = [by_group[g][par]['samples'] for g in names]
            best_values = [by_group[g][par]['best'] for g in names]
            xlbls = tick_labels
            width = 0.4

        # boxplot skips the empty entries (the gap)
        positions = np.arange(1, len(samples) + 1)
        box_data = [a if len(a) else [np.nan] for a in samples]
        boxes = ax.boxplot(box_data, positions=positions, widths=width, **_CAIXA)
        for patch in boxes['boxes']:
            patch.set_facecolor(_COR)
        ax.plot(positions, best_values, '*', markersize=9, color='green',
                linestyle='none', label='max-LL')

        if par == 'ic':
            ax.set_xticks(positions)
            ax.set_xticklabels(xlbls)
            # labels centered on each group
            n0 = len([p for p in by_group[names[0]] if p.startswith('ic')])
            ax.set_xticks([n0 / 2 + 0.5, n0 + 1 + (len(samples) - n0 - 1) / 2])
            ax.set_xticklabels(tick_labels, fontsize=12)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(xlbls, fontsize=12)
            r = next(x for x in rows if x['param'] == par)
            if np.isfinite(r['p']):
                star = ' *' if r['significant'] else ''
                ax.set_title(f"{par}   Mann-Whitney p = {r['p']:.2e}{star}",
                             fontsize=11)
        ax.set_ylabel(par)
        ax.grid(True, which='major', alpha=0.1)

    for i in range(len(panels), nRows * nCols):
        fig.delaxes(axes[i])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    if save:
        base = os.path.join(out_dir, f'boxplot_groups{model}')
        fig.savefig(base + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(base + '.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return rows


def print_comparison(rows):
    if not rows:
        return
    print(f"\n=== Mann-Whitney U between groups ({rows[0]['model']}) ===")
    print(f'{"param":<10s}{"median A":>13s}{"median B":>13s}{"U":>14s}'
          f'{"p":>12s}  sig')
    for r in rows:
        sig = '*' if r['significant'] else ''
        u = f"{r['U']:.0f}" if np.isfinite(r['U']) else '-'
        p = f"{r['p']:.3e}" if np.isfinite(r['p']) else '-'
        print(f"{r['param']:<10s}{r['median_a']:>13.4g}{r['median_b']:>13.4g}"
              f"{u:>14s}{p:>12s}  {sig}")
    print('  A = ' + rows[0]['group_a'] + ' | B = ' + rows[0]['group_b'])
    print('  optimistic p-values: MCMC samples are autocorrelated')


def run_group_comparison(results_dir='.', groups=DEFAULT_GROUPS,
                         models=DEFAULT_MODELS, mode='global', full_data=None,
                         save_csv=True, save_fig=True, print_table=True):
    """Runs the comparison for each model and writes the CSV file and figures."""
    if full_data is None:
        from data_loader import load_control_data
        full_data = load_control_data()

    all_rows = []
    for model in models:
        if save_fig:
            rows = plot_group_boxplots(results_dir, groups, model, mode,
                                       full_data, show=False, save=True)
        else:
            rows, _ = compare_groups(results_dir, groups, model, mode, full_data)
        if not rows:
            continue
        if print_table:
            print_comparison(rows)
        all_rows.extend(rows)

    if not all_rows:
        if print_table:
            print(f'[group_comparison] no results in {results_dir}')
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if save_csv:
        out = os.path.join(results_dir, 'Output_Calibration',
                           f'group_comparison_{mode}.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False, float_format='%.6g')
        if print_table:
            print(f'\nCSV: {out}')
    return df


def _cli():
    ap = argparse.ArgumentParser(
        description='Box plots and Mann-Whitney U test of the posteriors between groups.')
    ap.add_argument('results_dir', nargs='?', default='.')
    ap.add_argument('--mode', default='global',
                    choices=['global', 'mouse_specific'])
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS))
    ap.add_argument('--groups', default=','.join(DEFAULT_GROUPS))
    ap.add_argument('--no-csv', action='store_true')
    ap.add_argument('--no-fig', action='store_true')
    args = ap.parse_args()

    run_group_comparison(
        results_dir=args.results_dir, mode=args.mode,
        groups=args.groups.split(','), models=args.models.split(','),
        save_csv=not args.no_csv, save_fig=not args.no_fig, print_table=True)


if __name__ == '__main__':
    _cli()
