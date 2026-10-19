"""Observed versus predicted tumor volumes of a growth model calibrated separately to each control group
(thesis Figure 4.3), with the posterior median and 95% credible interval of the cohort CCC and PCC.

    python thesis_scatter_figure.py --model _log --out Output_Calibration/thesis_figures/fig3.png

The points are the predictions of the maximum-likelihood sample; CCC and PCC are summarized over
posterior draws.
"""
import argparse
import os
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import thesis_style as ts
from data_loader import load_control_data
from metrics import ccc, pcc
from models import _split_theta, solve_model

GROUPS = ['control_sensitive', 'control_resistant']


def predictions(theta, model, mice):
    """List of predicted volumes at the observation times, one array per mouse."""
    ode_args, ic_fn = _split_theta(theta, model, len(mice))
    return [solve_model(model, m['data'][:, 0], ode_args, ic_fn(i))[:, 0] for i, m in enumerate(mice)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='_log')
    ap.add_argument('--draws', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='Output_Calibration/thesis_figures/fig3.png')
    args = ap.parse_args()

    full_data = load_control_data()
    rng = np.random.default_rng(args.seed)
    ts.default_font(14)
    fig, axes = plt.subplots(1, 2, figsize=(22, 15 * (1.4 / (1 + 1 + 1.4))))
    for k, (ax, g) in enumerate(zip(axes, GROUPS)):
        mice = full_data[g]
        observed = [m['data'][:, 1] for m in mice]
        obs = np.concatenate(observed)
        chain = np.loadtxt(f'Output_Calibration/multi_chain_{g}{args.model}.gz', ndmin=2)
        best = np.load(f'Output_Calibration/multi_ll_pars_{g}{args.model}.npz', allow_pickle=True)['pars']
        draws = chain[rng.choice(len(chain), size=min(args.draws, len(chain)), replace=False)]
        values = np.array([(ccc(p, obs), pcc(p, obs))
                           for p in (np.concatenate(predictions(t, args.model, mice)) for t in draws)])
        c = tuple(np.percentile(values[:, 0], [50, 2.5, 97.5]))
        p = tuple(np.percentile(values[:, 1], [50, 2.5, 97.5]))
        print(f'{g}: CCC {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]')
        ts.scatter_panel(ax, observed, predictions(best, args.model, mice), string.ascii_uppercase[k], c, p,
                         style='inline')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print('written', args.out)


if __name__ == '__main__':
    main()
