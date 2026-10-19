"""Thesis figures of the immunotherapy-only cohort, de Pillis-Radunskaya formulation (Section 4.4).

    CALIB_GROWTH_SOURCE=fixed python thesis_figures.py [--figures fig10,fig11] [--out-dir Output_Calibration/thesis_figures]

fig10  posterior histograms of d, s and lambda (fig10a-c)
fig11  every mouse of the cohort with the global and the mouse-specific calibrations, and the scatter plot

CCC and PCC are summarized by the posterior median and 95% credible interval over posterior draws; the
credible band is the 2.5-97.5 percentiles of the trajectories of 200 draws at the observation times; the
solid line is the trajectory of the maximum-likelihood sample of the global calibration and the dashed line
that of the mouse-specific calibration (Output_Calibration_per_mouse).
"""
import argparse
import os
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import _path_setup  # noqa: F401
import thesis_style as ts
from growth_params import load_growth_params
from metrics import ccc, pcc

from bounds_imm import extract_treatment_params
from data_loader_imm import load_immuno_data
from immuno_models import solve_model

GROUP = 'io_sensitive'
MODEL = '_dePillisRadunskayaLaw'
DOSE = 1.0


def solve(theta, mouse_idx, mice, growth, smooth=False):
    params, n_slots = extract_treatment_params(theta, MODEL, len(mice), mouse_idx)
    m = mice[mouse_idx]
    r, cc = growth[GROUP]['r'], growth[GROUP]['cc']
    sol = solve_model(MODEL, m['data'][:, 0], (r, cc, *params), float(theta[n_slots + mouse_idx]),
                      m['treatment_days'], DOSE, 'smooth' if smooth else 'data')
    return (sol[:, 0], sol[:, 1]) if smooth else sol[:, 0]


PANEL_WIDTH = 1.76      # width of each panel of Table 4.14 on the page, in inches
PRIOR_BOUNDS = (0.01, 2.0)   # d, s and lambda share these bounds (bounds_imm)


def fig10(chain, best, labels, out_dir):
    panels = (('d', 'fig10a', r'$d$ [day$^{-1}$]'),
              ('s', 'fig10b', r'$s$ [(mm$^3$)$^{-\lambda}$]'),
              ('lam', 'fig10c', r'$\lambda$'))
    for name, stem, label in panels:
        i = labels.index(name)
        ts.posterior_histogram(chain[:, i], best[i], os.path.join(out_dir, stem + '.png'),
                               PANEL_WIDTH, label, bounds=PRIOR_BOUNDS)
        q = np.percentile(chain[:, i], [50, 2.5, 97.5])
        print(f'{name}: median {q[0]:.4g} [{q[1]:.4g}, {q[2]:.4g}], ML {best[i]:.4g}')


def fig11(chain, best, mice, growth, out_dir, draws, band_draws, seed):
    rng = np.random.default_rng(seed)
    thetas = chain[rng.choice(len(chain), size=min(draws, len(chain)), replace=False)]
    obs_all = np.concatenate([m['data'][:, 1] for m in mice])
    per_mouse = [[] for _ in mice]
    traj = [[] for _ in mice]
    cohort = []
    for k, theta in enumerate(thetas):
        preds = [solve(theta, i, mice, growth) for i in range(len(mice))]
        for i, m in enumerate(mice):
            per_mouse[i].append((ccc(preds[i], m['data'][:, 1]), pcc(preds[i], m['data'][:, 1])))
            if k < band_draws:
                traj[i].append(preds[i])
        cohort.append((ccc(np.concatenate(preds), obs_all), pcc(np.concatenate(preds), obs_all)))

    def summ(values):
        v = np.asarray(values)
        return [tuple(np.percentile(v[:, j], [50, 2.5, 97.5])) for j in (0, 1)]

    ts.times_font(18)
    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    gs.update(hspace=0.55)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.subplots_adjust(top=0.95)
    pos = axes[3].get_position()
    axes[3].set_position([pos.x0, pos.y0 - 0.2 * pos.height, pos.width, 1.4 * pos.height])
    tmax = max(m['data'][:, 0].max() for m in mice)
    best_pred = []
    for i, m in enumerate(mice):
        ms = np.load(os.path.join('Output_Calibration_per_mouse', f'll_pars_{GROUP}_{m["name"]}{MODEL}.npz'))['pars']
        r, cc = growth[GROUP]['r'], growth[GROUP]['cc']
        sol_ms = solve_model(MODEL, m['data'][:, 0], (r, cc, *ms[:-2]), float(ms[-2]), m['treatment_days'], DOSE,
                             'smooth')
        band = tuple(np.percentile(np.array(traj[i]), [97.5, 2.5], axis=0))
        ts.fit_panel(axes[i], m['data'], solve(best, i, mice, growth, smooth=True), 'Global', 'selected',
                     band=band, extra_curves=[(sol_ms[:, 0], sol_ms[:, 1], 'Mouse-specific', 'blue', 'dashed')],
                     treatments=[(m['treatment_days'], 'red')], xmax=tmax, legend=(i == 0))
        c, p = summ(per_mouse[i])
        ts.metrics_text(axes[i], 'selected', string.ascii_uppercase[i], c, p)
        ts.volume_yaxis(axes[i], 2000)
        best_pred.append(solve(best, i, mice, growth))
        print(f'{m["name"]}: CCC {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]')
    c, p = summ(cohort)
    print(f'cohort: CCC {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]')
    ts.scatter_panel(axes[3], [m['data'][:, 1] for m in mice], best_pred, 'D', c, p, style='rtio')
    fig.savefig(os.path.join(out_dir, 'fig11.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figures', default='fig10,fig11')
    ap.add_argument('--draws', type=int, default=500)
    ap.add_argument('--band-draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='Output_Calibration/thesis_figures')
    args = ap.parse_args()
    which = args.figures.split(',')
    os.makedirs(args.out_dir, exist_ok=True)
    chain = np.loadtxt(f'Output_Calibration/multi_chain_{GROUP}{MODEL}.gz', ndmin=2)
    npz = np.load(f'Output_Calibration/multi_ll_pars_{GROUP}{MODEL}.npz', allow_pickle=True)
    best, labels = np.asarray(npz['pars'], float), [str(x) for x in npz['labels']]
    if 'fig10' in which:
        fig10(chain, best, labels, args.out_dir)
    if 'fig11' in which:
        mice = load_immuno_data()[GROUP]
        # growth parameters of the thesis (Section 4.2), independently of CALIB_GROWTH_SOURCE
        growth = load_growth_params([GROUP], source='fixed')
        fig11(chain, best, mice, growth, args.out_dir, args.draws, args.band_draws, args.seed)
    print('figures written to', args.out_dir)


if __name__ == '__main__':
    main()
