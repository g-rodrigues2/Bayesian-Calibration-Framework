"""Thesis figures of the radiotherapy parameter sharing scenarios (Section 4.3).

    CALIB_GROWTH_SOURCE=fixed CALIB_DOSE_DAYS=0,1,2,3,4,5 python thesis_figures.py \
        [--figures fig7,fig8,fig9,figS2] [--out-dir Output_Calibration/thesis_figures]

fig7          posterior of alpha, beta and delta in the eight scenarios (violins). Scenario 2 is read from
              Output_Calibration (converged chain), the other scenarios from Output_Calibration/legacy
              (original chains converted by legacy_scenario_chains.py)
fig8          posterior histograms of alpha, beta_s, delta and beta_r under scenario 2 (fig8a-d)
fig9          best and worst mouse of each phenotype and scatter plots under scenario 2
figS2         every mouse of each phenotype (figS2_1 sensitive, figS2_2 resistant)

CCC and PCC are summarized by the posterior median and 95% credible interval over posterior draws;
credible bands are the 2.5-97.5 percentiles of the trajectories of 200 draws at the observation times;
the solid line is the trajectory of the maximum-likelihood sample.
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

from data_loader_rad import load_radiation_data, parse_dose_days_env
from inference_scen_rad import DOSE
from radiation_models import solve_model_dispatch
from scenarios_rad import MODEL_PARAM_NAMES, SCENARIOS, WINNER_MODEL, count_model_params, ic_slice_for_group, \
    split_model_args_for_group

GROUPS = ['radiation_sensitive', 'radiation_resistant']
MODEL = WINNER_MODEL


def load(scenario, legacy_dir):
    base = 'Output_Calibration' if scenario == 2 else legacy_dir
    stem = f'{MODEL}_V{scenario}'
    chain = np.loadtxt(os.path.join(base, f'multi_chain_{stem}.gz'), ndmin=2)
    npz = np.load(os.path.join(base, f'multi_ll_pars_{stem}.npz'), allow_pickle=True)
    return chain, np.asarray(npz['pars'], float), [str(x) for x in npz['labels']]


def simulate(theta, full_data, growth, smooth=False):
    sdef = SCENARIOS[MODEL][2]
    n_model = count_model_params(sdef, GROUPS)
    out = {}
    for g in GROUPS:
        args, _ = split_model_args_for_group(theta, sdef, MODEL_PARAM_NAMES[MODEL], GROUPS, g)
        T0s = ic_slice_for_group(theta, n_model, full_data, GROUPS, g)
        r, cc = growth[g]['r'], growth[g]['cc']
        sols = []
        for i, m in enumerate(full_data[g]):
            sol = solve_model_dispatch(MODEL, m['data'][:, 0], (r, cc, *args), float(T0s[i]), m['treatment_days'],
                                       DOSE, 'smooth' if smooth else 'data')
            sols.append((sol[:, 0], sol[:, 1]) if smooth else sol[:, 0])
        out[g] = sols
    return out


def posterior_summaries(chain, full_data, growth, draws, band_draws, seed):
    rng = np.random.default_rng(seed)
    thetas = chain[rng.choice(len(chain), size=min(draws, len(chain)), replace=False)]
    per_mouse = {g: [[] for _ in full_data[g]] for g in GROUPS}
    cohort = {g: [] for g in GROUPS}
    traj = {g: [[] for _ in full_data[g]] for g in GROUPS}
    obs_all = {g: np.concatenate([m['data'][:, 1] for m in full_data[g]]) for g in GROUPS}
    for k, theta in enumerate(thetas):
        pred = simulate(theta, full_data, growth)
        for g in GROUPS:
            for i, m in enumerate(full_data[g]):
                per_mouse[g][i].append((ccc(pred[g][i], m['data'][:, 1]), pcc(pred[g][i], m['data'][:, 1])))
                if k < band_draws:
                    traj[g][i].append(pred[g][i])
            model_all = np.concatenate(pred[g])
            cohort[g].append((ccc(model_all, obs_all[g]), pcc(model_all, obs_all[g])))

    def summ(values):
        v = np.asarray(values)
        return [tuple(np.percentile(v[:, j], [50, 2.5, 97.5])) for j in (0, 1)]

    mice = {g: [summ(per_mouse[g][i]) for i in range(len(full_data[g]))] for g in GROUPS}
    bands = {g: [tuple(np.percentile(np.array(traj[g][i]), [97.5, 2.5], axis=0)) for i in range(len(full_data[g]))]
             for g in GROUPS}
    return mice, {g: summ(cohort[g]) for g in GROUPS}, bands


def fig7(legacy_dir, out_dir):
    ts.times_font(20)
    fig, axes = plt.subplots(3, 1, figsize=(22, 15))
    for k, (base, ylabel) in enumerate([('alpha_rad', r'$\alpha\,$ (Gy$\times$day)$^{-1}$'),
                                        ('beta_rad', r'$\beta\,$ (day$^{-1})$'),
                                        ('delay', r'$\delta\,$ (day)')]):
        entries = []
        for scenario in range(1, 9):
            chain, best, labels = load(scenario, legacy_dir)
            if f'{base}_global' in labels:
                i = labels.index(f'{base}_global')
                entries.append(('global', chain[:, i], best[i]))
            else:
                i, j = labels.index(f'{base}_sensitive'), labels.index(f'{base}_resistant')
                entries.append(('group', (chain[:, i], chain[:, j]), (best[i], best[j])))
        ts.scenario_violins(axes[k], entries, ylabel, string.ascii_uppercase[k])
        axes[k].set_xlabel('Scenario' if k == 2 else '', fontsize=20)
    ts.violin_legend(axes[0])
    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(out_dir, 'fig7.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


PANEL_WIDTH = 1.82      # width of each panel of Table 4.11 on the page, in inches


def fig8(out_dir):
    chain, best, labels = load(2, None)
    panels = (('alpha_rad_global', 'fig8a', r'$\alpha$ [(Gy$\times$day)$^{-1}$]'),
              ('beta_rad_sensitive', 'fig8b', r'$\beta_s$ [day$^{-1}$]'),
              ('delay_global', 'fig8c', r'$\delta$ [day]'),
              ('beta_rad_resistant', 'fig8d', r'$\beta_r$ [day$^{-1}$]'))
    for name, stem, label in panels:
        i = labels.index(name)
        ts.posterior_histogram(chain[:, i], best[i], os.path.join(out_dir, stem + '.png'),
                               PANEL_WIDTH, label)
        q = np.percentile(chain[:, i], [50, 2.5, 97.5])
        print(f'{name}: median {q[0]:.4g} [{q[1]:.4g}, {q[2]:.4g}], ML {best[i]:.4g}')
    bs, br = chain[:, labels.index('beta_rad_sensitive')], chain[:, labels.index('beta_rad_resistant')]
    q = np.percentile(br / bs, [50, 2.5, 97.5])
    print(f'beta_r/beta_s: median {q[0]:.3f} [{q[1]:.3f}, {q[2]:.3f}], P(beta_r > beta_s) = {np.mean(br > bs):.4f}')


def fits(full_data, growth, out_dir, draws, band_draws, seed, which):
    chain, best, _ = load(2, None)
    mice, cohort, bands = posterior_summaries(chain, full_data, growth, draws, band_draws, seed)
    best_data = simulate(best, full_data, growth)
    best_smooth = simulate(best, full_data, growth, smooth=True)
    tmax = {g: max(m['data'][:, 0].max() for m in full_data[g]) for g in GROUPS}
    for g in GROUPS:
        for m, s in zip(full_data[g], mice[g]):
            print(f'{g} {m["name"]}: CCC {s[0][0]:.3f} [{s[0][1]:.3f}, {s[0][2]:.3f}]')
        c, p = cohort[g]
        print(f'{g} cohort: CCC {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]  PCC {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]')



    if 'fig9' in which:
        ts.times_font(20)
        fig = plt.figure(figsize=(22, 15))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.4])
        gs.update(hspace=0.35)
        axes_ts = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        axes_sc = [fig.add_subplot(gs[2, j]) for j in range(2)]
        plotted = []
        for row, g in enumerate(GROUPS):
            order = np.argsort([s[0][0] for s in mice[g]])
            for col, i in enumerate([order[-1], order[0]]):
                ax = axes_ts[2 * row + col]
                m = full_data[g][i]
                ts.fit_panel(ax, m['data'], best_smooth[g][i], 'Scenario 2', 'best_worst', band=bands[g][i],
                             treatments=[(m['treatment_days'], 'green')], xmax=max(tmax.values()),
                             legend=(row == 0 and col == 0))
                ts.metrics_text(ax, 'best_worst', string.ascii_uppercase[2 * row + col], *mice[g][i])
                plotted += [m['data'][:, 1], best_data[g][i]]
        top = ts.rounded_thousands(*plotted)
        for ax in axes_ts:
            ts.volume_yaxis(ax, top)
        for col, g in enumerate(GROUPS):
            ts.scatter_panel(axes_sc[col], [m['data'][:, 1] for m in full_data[g]], best_data[g],
                             string.ascii_uppercase[4 + col], *cohort[g], style='best_worst')
        fig.tight_layout()
        fig.subplots_adjust(top=1)
        fig.savefig(os.path.join(out_dir, 'fig9.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

    if 'figS2' in which:
        for g, name in zip(GROUPS, ('figS2_1', 'figS2_2')):
            ts.times_font(20)
            n = len(full_data[g])
            ncols, nrows = 3, int(np.ceil(n / 3))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 5 * nrows), squeeze=False)
            axes = axes.ravel()
            for i, m in enumerate(full_data[g]):
                ts.fit_panel(axes[i], m['data'], best_smooth[g][i], 'Scenario 2', 'all_mice', band=bands[g][i],
                             treatments=[(m['treatment_days'], 'green')], xmax=tmax[g], legend=(i == 0))
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
    ap.add_argument('--figures', default='fig7,fig8,fig9,figS2')
    ap.add_argument('--legacy-dir', default='Output_Calibration/legacy')
    ap.add_argument('--draws', type=int, default=500, help='posterior draws for CCC/PCC')
    ap.add_argument('--band-draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='Output_Calibration/thesis_figures')
    args = ap.parse_args()
    which = args.figures.split(',')
    os.makedirs(args.out_dir, exist_ok=True)
    if 'fig7' in which:
        fig7(args.legacy_dir, args.out_dir)
    if 'fig8' in which:
        fig8(args.out_dir)
    if {'fig9', 'figS2'} & set(which):
        full_data = load_radiation_data(data_dir=_path_setup.DATA_DIR,
                                        dose_days=parse_dose_days_env(os.environ.get('CALIB_DOSE_DAYS')))
        # growth parameters of the thesis (Section 4.2), independently of CALIB_GROWTH_SOURCE
        growth = load_growth_params(GROUPS, source='fixed')
        fits(full_data, growth, args.out_dir, args.draws, args.band_draws, args.seed, which)
    print('figures written to', args.out_dir)


if __name__ == '__main__':
    main()
