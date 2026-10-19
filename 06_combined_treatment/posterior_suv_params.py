"""Posterior distributions of the SUV-response parameters a and b of the linear formulation.

    python posterior_suv_params.py
    python posterior_suv_params.py --backend <reinitiate.h5>     # any emcee HDF5 backend

Reads the chain written by `calibrate.py suv` (columns a, b, ic_1..ic_N, std), or an emcee HDF5
backend, from which the first quarter of the steps and the stuck walkers (walkers.drop_stuck) are
removed. Writes a table with
the maximum-likelihood sample, posterior mean, standard deviation, median and 95% credible interval
of a, b and of the lysis coefficient phi(SUV) = a SUV + b over the cohort SUV range, and the
marginal posterior histograms.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _path_setup  # noqa: F401
from walkers import drop_stuck

from data_loader_ri import load_data

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = 'rad+io_sensitive_SUVLinear'
OUT = os.path.join(HERE, 'Output_Calibration', 'suv')


def summary(name, samples, map_value):
    lo, med, hi = np.percentile(samples, [2.5, 50, 97.5])
    return dict(quantity=name, map=map_value, mean=samples.mean(), sd=samples.std(ddof=1),
                median=med, ci95_low=lo, ci95_high=hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chain', default=os.path.join(OUT, f'chain_{STEM}.gz'))
    ap.add_argument('--estimate', default=os.path.join(OUT, f'estimate_{STEM}.npz'))
    ap.add_argument('--backend', default=None, help='emcee HDF5 backend (overrides --chain and --estimate)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    if args.backend:
        import emcee
        backend = emcee.backends.HDFBackend(args.backend, read_only=True)
        full, lp = backend.get_chain(), backend.get_log_prob()
        best = full.reshape(-1, full.shape[2])[np.argmax(lp)]
        kept, _, stuck = drop_stuck(full[len(full) // 4:], lp[len(lp) // 4:])
        chain = kept.reshape(-1, kept.shape[2])
        print(f'{int(stuck.sum())} of {len(stuck)} walkers removed as stuck')
    else:
        chain = np.loadtxt(args.chain, ndmin=2)
        best = np.load(args.estimate)['pars']
    suv = np.array([m['SUV'] for m in load_data()['rad+io_sensitive']])
    a, b = chain[:, 0], chain[:, 1]
    rows = [summary('a', a, best[0]), summary('b', b, best[1])]
    for label, value in (('min', suv.min()), ('max', suv.max())):
        rows.append(summary(f'phi(SUV={value:.3f}, cohort {label})', a * value + b, best[0] * value + best[1]))
    table = pd.DataFrame(rows)
    table['p_positive'] = [np.mean(a > 0), np.mean(b > 0)] + [np.mean(a * v + b > 0) for v in (suv.min(), suv.max())]
    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, 'posterior_suv_params.csv'), index=False, float_format='%.5g')
    print(f'{len(chain)} posterior samples')
    print(table.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, samples, value, name in zip(axes, (a, b), best[:2], ('a', 'b')):
        ax.hist(samples, bins=60, density=True, color='#9ecae1', edgecolor='white')
        ax.axvline(value, color='#08519c', lw=2, label='maximum-likelihood sample')
        ax.axvline(np.median(samples), color='k', ls='--', lw=1.2, label='posterior median')
        ax.set_xlabel(name)
        ax.set_ylabel('posterior density')
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'posterior_suv_params.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
