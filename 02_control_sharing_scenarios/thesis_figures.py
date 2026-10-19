"""Thesis figures of the growth-parameter sharing scenarios (Section 4.2).

    python thesis_figures.py [--figures fig4,fig5,fig6,figS1] [--out-dir Output_Calibration/thesis_figures]

fig4   posterior of r and K in the four sharing scenarios (violins)
fig5   posterior histograms of r_s, K and r_r under scenario 2 (fig5a-c)
fig6   best and worst mouse of each phenotype and scatter plots under scenario 2
figS1  every mouse of each phenotype under scenario 2 (figS1_1 sensitive, figS1_2 resistant)

Per-mouse and cohort CCC and PCC are summarized by the posterior median and 95% credible interval over
posterior draws; credible bands are the 2.5-97.5 percentiles of the trajectories of 200 draws at the
observation times; the solid line is the trajectory of the maximum-likelihood sample.
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
from data_loader import load_control_data
from metrics import ccc, pcc
from models import solve_model
from scenarios import MODEL_PARAM_NAMES, SCENARIOS, count_model_params, ic_slice_for_group, split_model_args_for_group

GROUPS = ['control_sensitive', 'control_resistant']
MODEL = '_log'


def load(scenario):
    stem = f'{MODEL}_s{scenario}'
    chain = np.loadtxt(f'Output_Calibration/multi_chain_{stem}.gz', ndmin=2)
    npz = np.load(f'Output_Calibration/multi_ll_pars_{stem}.npz', allow_pickle=True)
    return chain, np.asarray(npz['pars'], float), [str(x) for x in npz['labels']]


def simulate(theta, scenario, full_data, smooth=False):
    """{group: [per mouse]} volumes at the observation times, or (t, T) on the 0.1-day grid."""
    sdef = SCENARIOS[MODEL][scenario]
    n_model = count_model_params(sdef, GROUPS)
    out = {}
    for g in GROUPS:
        args, _ = split_model_args_for_group(theta, sdef, MODEL_PARAM_NAMES[MODEL], GROUPS, g)
        T0s = ic_slice_for_group(theta, n_model, full_data, GROUPS, g)
        sols = []
        for i, m in enumerate(full_data[g]):
            sol = solve_model(MODEL, m['data'][:, 0], args, float(T0s[i]), 'smooth' if smooth else 'data')
            sols.append((sol[:, 0], sol[:, 1]) if smooth else sol[:, 0])
        out[g] = sols
    return out


def posterior_summaries(chain, scenario, full_data, draws, band_draws, seed):
    """Per-mouse CCC/PCC summaries, cohort summaries and credible bands at the observation times."""
    rng = np.random.default_rng(seed)
    thetas = chain[rng.choice(len(chain), size=min(draws, len(chain)), replace=False)]
    per_mouse = {g: [[] for _ in full_data[g]] for g in GROUPS}
    cohort = {g: [] for g in GROUPS}
    traj = {g: [[] for _ in full_data[g]] for g in GROUPS}
    for k, theta in enumerate(thetas):
        pred = simulate(theta, scenario, full_data)
        for g in GROUPS:
            for i, m in enumerate(full_data[g]):
                obs = m['data'][:, 1]
                per_mouse[g][i].append((ccc(pred[g][i], obs), pcc(pred[g][i], obs)))
                if k < band_draws:
                    traj[g][i].append(pred[g][i])
            cohort[g].append((ccc(np.concatenate(pred[g]), np.concatenate([m['data'][:, 1] for m in full_data[g]])),
                              pcc(np.concatenate(pred[g]), np.concatenate([m['data'][:, 1] for m in full_data[g]]))))

    def summ(values):
        v = np.asarray(values)
        return [tuple(np.percentile(v[:, j], [50, 2.5, 97.5])) for j in (0, 1)]

    mice = {g: [summ(per_mouse[g][i]) for i in range(len(full_data[g]))] for g in GROUPS}
    bands = {g: [tuple(np.percentile(np.array(traj[g][i]), [97.5, 2.5], axis=0)) for i in range(len(full_data[g]))]
             for g in GROUPS}
    return mice, {g: summ(cohort[g]) for g in GROUPS}, bands


def fig4(full_data, out_dir):
    ts.times_font(20)
    fig, axes = plt.subplots(1, 2, figsize=(22, 7.5))
    for k, (base, ylabel, scale) in enumerate([('r', r'$r\,$ (day$^{-1}$)', 1.0),
                                                ('cc', r'$K\,$ ($\times10^3$ mm$^3$)', 1e-3)]):
        entries = []
        for scenario in (1, 2, 3, 4):
            chain, best, labels = load(scenario)
            if f'{base}_global' in labels:
                i = labels.index(f'{base}_global')
                entries.append(('global', chain[:, i] * scale, best[i] * scale))
            else:
                i, j = labels.index(f'{base}_sensitive'), labels.index(f'{base}_resistant')
                entries.append(('group', (chain[:, i] * scale, chain[:, j] * scale), (best[i] * scale, best[j] * scale)))
        ts.scenario_violins(axes[k], entries, ylabel, string.ascii_uppercase[k])
        axes[k].set_xlabel('Scenario', fontsize=20)
    ts.violin_legend(axes[0])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


PANEL_WIDTH = 1.42      # width of each panel of Table 4.5 on the page, in inches


def fig5(out_dir):
    chain, best, labels = load(2)
    rows = []
    panels = (('r_sensitive', 'fig5a', r'$r_s$ [day$^{-1}$]'),
              ('cc_global', 'fig5b', r'$K$ [mm$^3$]'),
              ('r_resistant', 'fig5c', r'$r_r$ [day$^{-1}$]'))
    for name, stem, label in panels:
        i = labels.index(name)
        ts.posterior_histogram(chain[:, i], best[i], os.path.join(out_dir, stem + '.png'),
                               PANEL_WIDTH, label)
        q = np.percentile(chain[:, i], [50, 2.5, 97.5])
        rows.append(f'{name}: median {q[0]:.4g} [{q[1]:.4g}, {q[2]:.4g}], ML {best[i]:.4g}')
    ratio = chain[:, labels.index('r_resistant')] / chain[:, labels.index('r_sensitive')]
    q = np.percentile(ratio, [50, 2.5, 97.5])
    rows.append(f'r_r/r_s: median {q[0]:.3f} [{q[1]:.3f}, {q[2]:.3f}], P(r_r > r_s) = {np.mean(ratio > 1):.4f}')
    print('\n'.join(rows))


def fig6_and_S1(full_data, out_dir, draws, band_draws, seed, which):
    chain, best, _ = load(2)
    mice, cohort, bands = posterior_summaries(chain, 2, full_data, draws, band_draws, seed)
    best_data = simulate(best, 2, full_data)
    best_smooth = simulate(best, 2, full_data, smooth=True)
    tmax = {g: max(m['data'][:, 0].max() for m in full_data[g]) for g in GROUPS}
    for g in GROUPS:
        for m, s in zip(full_data[g], mice[g]):
            print(f'{g} {m["name"]}: CCC {s[0][0]:.3f} [{s[0][1]:.3f}, {s[0][2]:.3f}]')
        c, p = cohort[g]
        print(f'{g} cohort: CCC {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]')

    if 'fig6' in which:
        ts.times_font(20)
        fig = plt.figure(figsize=(22, 15))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.4])
        gs.update(hspace=0.35)
        axes_ts = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        axes_sc = [fig.add_subplot(gs[2, j]) for j in range(2)]
        for row, g in enumerate(GROUPS):
            order = np.argsort([s[0][0] for s in mice[g]])
            for col, i in enumerate([order[-1], order[0]]):
                ax = axes_ts[2 * row + col]
                m = full_data[g][i]
                ts.fit_panel(ax, m['data'], best_smooth[g][i], 'Scenario 2', 'best_worst', band=bands[g][i],
                             xmax=max(tmax.values()), legend=(row == 0 and col == 0))
                ts.metrics_text(ax, 'best_worst', string.ascii_uppercase[2 * row + col], *mice[g][i])
                # each panel has its own volume scale, so that small tumors remain readable
                ts.volume_yaxis_fitted(ax, m['data'][:, 1], best_data[g][i])
                legend = ax.get_legend()
                if legend is not None:                  # growth curves leave the lower right corner empty
                    size = legend.get_texts()[0].get_fontsize()
                    handles, labels = ax.get_legend_handles_labels()
                    legend.remove()
                    ax.legend(handles, labels, loc='lower right', fontsize=size)
        for col, g in enumerate(GROUPS):
            ts.scatter_panel(axes_sc[col], [m['data'][:, 1] for m in full_data[g]], best_data[g],
                             string.ascii_uppercase[4 + col], *cohort[g], style='best_worst')
        fig.tight_layout()
        fig.subplots_adjust(top=1)
        fig.savefig(os.path.join(out_dir, 'fig6.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

    if 'figS1' in which:
        for g, name in zip(GROUPS, ('figS1_1', 'figS1_2')):
            ts.times_font(20)
            n = len(full_data[g])
            ncols, nrows = 3, int(np.ceil(n / 3))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 5 * nrows), squeeze=False)
            axes = axes.ravel()
            for i, m in enumerate(full_data[g]):
                ts.fit_panel(axes[i], m['data'], best_smooth[g][i], 'Scenario 2', 'all_mice', band=bands[g][i],
                             xmax=tmax[g], legend=(i == 0))
                ts.metrics_text(axes[i], 'all_mice', string.ascii_uppercase[i], *mice[g][i])
            top = ts.rounded_thousands(*[m['data'][:, 1] for m in full_data[g]], *best_data[g])
            for ax in axes[:n]:
                ts.volume_yaxis(ax, top)
            for ax in axes[n:]:
                fig.delaxes(ax)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figures', default='fig4,fig5,fig6,figS1')
    ap.add_argument('--draws', type=int, default=500, help='posterior draws for CCC/PCC')
    ap.add_argument('--band-draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='Output_Calibration/thesis_figures')
    args = ap.parse_args()
    which = args.figures.split(',')
    os.makedirs(args.out_dir, exist_ok=True)
    full_data = load_control_data()
    if 'fig4' in which:
        fig4(full_data, args.out_dir)
    if 'fig5' in which:
        fig5(args.out_dir)
    if 'fig6' in which or 'figS1' in which:
        fig6_and_S1(full_data, args.out_dir, args.draws, args.band_draws, args.seed, which)
    print('figures written to', args.out_dir)


if __name__ == '__main__':
    main()
