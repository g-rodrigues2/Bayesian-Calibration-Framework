"""Walker traces and convergence diagnostics of the chains reported in Chapter 4 (Appendix).

    python thesis_chain_traces.py [--out-dir Output_Calibration/thesis_figures/chains]

Chains of the selected models (control growth per phenotype, growth sharing scenario 2, radiotherapy
sharing scenario 2, de Pillis-Radunskaya immunotherapy, prediction of the combined cohort with fixed
parameters, SUV-based lysis) and of the leave-one-out analysis (training folds and held-out initial
volumes of both formulations) are read from the HDF5 backends of each stage. For every chain the
figure shows the traces of all walkers (black: retained; orange: removed as stuck), with the end of the
burn-in period (first quarter of the steps) as a dashed red line. The diagnostics of the post-burn-in
chain after stuck-walker removal are written to chain_diagnostics.csv and chain_diagnostics.tex:
integrated autocorrelation time, chain length over autocorrelation time and split R-hat (walkers as
chains), with the worst value over all parameters.
"""
import argparse
import glob
import os

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import thesis_style as ts
from run_until_converged import split_rhat
from walkers import find_stuck_walkers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SYMBOL = {'r': '$r$', 'cc': '$K$', 'r_sensitive': '$r_s$', 'r_resistant': '$r_r$', 'cc_global': '$K$',
          'alpha_rad_global': r'$\alpha$', 'beta_rad_sensitive': r'$\beta_s$', 'beta_rad_resistant': r'$\beta_r$',
          'delay_global': r'$\delta$', 'd': '$d$', 's': '$s$', 'lam': r'$\lambda$', 'a': '$a$', 'b': '$b$',
          'std': r'$\sigma$'}

# key, description, backend (relative to the tnbc folder), estimate with parameter labels, log-sampled parameters
SELECTED = [
    ('control_sensitive', 'Logistic growth, radiation-sensitive control group (global calibration)',
     '01_control_growth/reinitiate_files/reinitiate_control_sensitive_log.h5',
     '01_control_growth/Output_Calibration/multi_ll_pars_control_sensitive_log.npz', ()),
    ('control_resistant', 'Logistic growth, radiation-resistant control group (global calibration)',
     '01_control_growth/reinitiate_files/reinitiate_control_resistant_log.h5',
     '01_control_growth/Output_Calibration/multi_ll_pars_control_resistant_log.npz', ()),
    ('growth_sharing', 'Logistic growth, sharing scenario 2 (both control groups)',
     '02_control_sharing_scenarios/reinitiate_files/reinitiate__log_s2.h5',
     '02_control_sharing_scenarios/Output_Calibration/multi_ll_pars__log_s2.npz', ()),
    ('radiotherapy_sharing', 'Delayed exponential decay, radiotherapy sharing scenario 2',
     '04_radiotherapy_sharing_scenarios/reinitiate_files/reinitiate__ExponentialDecayDelay_V2.h5',
     '04_radiotherapy_sharing_scenarios/Output_Calibration/multi_ll_pars__ExponentialDecayDelay_V2.npz', ()),
    ('immunotherapy', 'de Pillis-Radunskaya law, immunotherapy-only cohort ($d$ and $s$ sampled on the log scale)',
     '05_immunotherapy/reinitiate_files/reinitiate_io_sensitive_dePillisRadunskayaLaw_log-d-s.h5',
     '05_immunotherapy/Output_Calibration/multi_ll_pars_io_sensitive_dePillisRadunskayaLaw.npz', ('d', 's')),
    ('prediction', 'Radiotherapy + immunotherapy cohort, fixed parameters (initial volumes and std)',
     '06_combined_treatment/reinitiate_files/rad+io_sensitive_RadImmunoFixed.h5', None, ()),
    ('suv', 'Radiotherapy + immunotherapy cohort, linear SUV-based lysis',
     '06_combined_treatment/reinitiate_files/rad+io_sensitive_SUVLinear.h5',
     '06_combined_treatment/Output_Calibration/suv/estimate_rad+io_sensitive_SUVLinear.npz', ()),
]


def read_backend(path):
    backend = emcee.backends.HDFBackend(path, read_only=True)
    return backend.get_chain(), backend.get_log_prob()


def analyse(chain, log_prob):
    """Burn-in, stuck mask and diagnostics of the post-burn-in chain without stuck walkers."""
    steps = chain.shape[0]
    burn = steps // 4
    stuck = find_stuck_walkers(log_prob[burn:])
    post = chain[burn:, ~stuck, :]
    tau = emcee.autocorr.integrated_time(post, quiet=True)
    rhat = np.array([split_rhat(post[:, :, i]) for i in range(post.shape[2])])
    return burn, stuck, dict(steps=steps, walkers=chain.shape[1], stuck=int(stuck.sum()), tau_max=float(tau.max()),
                             steps_over_tau_min=float((steps - burn) / tau.max()), rhat_max=float(rhat.max()))


def labels_of(npz_path, ndim, n_mice=None):
    if npz_path is not None:
        return [str(x) for x in np.load(os.path.join(ROOT, npz_path), allow_pickle=True)['labels']]
    return [f'ic{i + 1}' for i in range(ndim - 1)] + ['std']


def display(label):
    if label.startswith('ic'):
        digits = ''.join(ch for ch in label if ch.isdigit())
        return rf'$T_{{0,{digits}}}$' if digits else r'$T_0$'
    return SYMBOL.get(label, label)


def selected_figures(out_dir, rows):
    for key, description, backend, npz, logged in SELECTED:
        path = os.path.join(ROOT, backend)
        if not os.path.exists(path):
            print(f'[skip] {key}: {backend} not found')
            continue
        chain, log_prob = read_backend(path)
        burn, stuck, diag = analyse(chain, log_prob)
        labels = labels_of(npz, chain.shape[2])
        shown = [i for i, lab in enumerate(labels) if not lab.startswith('ic')] or list(range(len(labels)))
        if key == 'prediction':
            shown = list(range(len(labels)))
        plot = chain[:, :, shown].copy()
        for j, i in enumerate(shown):
            if labels[i] in logged:
                plot[:, :, j] = np.exp(plot[:, :, j])
        ts.trace_figure(plot, [display(labels[i]) for i in shown], os.path.join(out_dir, f'trace_{key}.png'), burn,
                        removed=stuck, ncols=4 if len(shown) > 6 else 1)
        rows.append(dict(chain=key, description=description, **diag))
        print(f'{key}: {diag}')


def grid_figure(entries, path, label, ncols=3):
    """One panel per chain (leave-one-out folds): traces of one parameter."""
    ts.times_font(13)
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.3 * nrows), squeeze=False)
    for ax, (title, series, burn, stuck) in zip(axes.ravel(), entries):
        stride = max(1, series.shape[0] // 2000)
        x = np.arange(0, series.shape[0], stride)
        ax.plot(x, series[::stride, ~stuck], color='black', alpha=0.3, linewidth=0.4)
        if stuck.any():
            ax.plot(x, series[::stride, stuck], color='darkorange', alpha=0.7, linewidth=0.5)
        ax.axvline(burn, color='red', linestyle='--', linewidth=1)
        ax.set_xlim(0, series.shape[0])
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(label, fontsize=13)
        ax.tick_params(labelsize=10)
    for ax in axes.ravel()[len(entries):]:
        ax.remove()
    for ax in axes[-1]:
        ax.set_xlabel('Step number', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def loo_figures(out_dir, rows):
    reinit = os.path.join(ROOT, '06_combined_treatment', 'reinitiate_files')
    for model, name, params in (('_SUVLinear', 'with SUV', [(0, '$a$'), (1, '$b$'), (-1, r'$\sigma$')]),
                                ('_RadImmunoFixed', 'without SUV', [(-1, r'$\sigma$')])):
        folds = sorted(glob.glob(os.path.join(reinit, f'rad+io_sensitive_EXCL-*{model}.h5')))
        heldout = sorted(glob.glob(os.path.join(reinit, f'rad+io_sensitive_PRED-*{model}.h5')))
        fold_entries = {p: [] for _, p in params}
        ic_entries = []
        for kind, files in (('EXCL', folds), ('PRED', heldout)):
            for path in files:
                mouse = os.path.basename(path).split(f'_{kind}-')[1].split(model)[0]
                title = f'{mouse.replace("c0_m", "C0 M")}'
                chain, log_prob = read_backend(path)
                burn, stuck, diag = analyse(chain, log_prob)
                what = 'training fold without' if kind == 'EXCL' else 'held-out initial volume of'
                rows.append(dict(chain=f'loo{model}_{kind}_{mouse}', description=f'Leave-one-out {name}, {what} {mouse}',
                                 **diag))
                if kind == 'EXCL':
                    for i, label in params:
                        fold_entries[label].append((f'Fold without {title}', chain[:, :, i], burn, stuck))
                else:
                    ic_entries.append((title, chain[:, :, 0], burn, stuck))
        tag = 'suv' if model == '_SUVLinear' else 'nosuv'
        for label, entries in fold_entries.items():
            stem = {'$a$': 'a', '$b$': 'b', r'$\sigma$': 'std'}[label]
            grid_figure(entries, os.path.join(out_dir, f'trace_loo_{tag}_folds_{stem}.png'), label)
        grid_figure(ic_entries, os.path.join(out_dir, f'trace_loo_{tag}_heldout_ic.png'), r'$T_0$')
        print(f'leave-one-out {name}: {len(folds)} folds, {len(heldout)} held-out chains')


def _thousands(n):
    return f'{int(n):,}'.replace(',', '{,}')


def latex_selected(table):
    """Rows (chain, steps, walkers, stuck, tau_max, N/tau, R-hat) of the selected models."""
    lines = []
    for r in table[~table['chain'].str.startswith('loo')].itertuples():
        lines.append(f'{r.description} & {_thousands(r.steps)} & {r.walkers} & {r.stuck} & {r.tau_max:.0f} & '
                     f'{r.steps_over_tau_min:.1f} & {r.rhat_max:.3f} \\\\')
    return '\n'.join(lines)


def latex_loo(table):
    """One row per held-out mouse: steps, N/tau and R-hat of its four chains (stuck walkers of the SUV folds)."""
    loo = table[table['chain'].str.startswith('loo')]
    cells = {}
    for r in loo.itertuples():
        rest = r.chain[len('loo'):]
        kind = 'EXCL' if '_EXCL_' in rest else 'PRED'
        model, mouse = rest.split(f'_{kind}_')
        cells[(mouse, model, kind)] = r
    mice = sorted({k[0] for k in cells}, key=lambda m: int(m.split('_m')[1]))
    lines = []
    for mouse in mice:
        parts = [mouse.replace('c0_m', 'C0 M')]
        for model, kind in (('_SUVLinear', 'EXCL'), ('_RadImmunoFixed', 'EXCL'), ('_SUVLinear', 'PRED'),
                            ('_RadImmunoFixed', 'PRED')):
            r = cells[(mouse, model, kind)]
            parts.append(_thousands(r.steps))
            if (model, kind) == ('_SUVLinear', 'EXCL'):
                parts.append(str(r.stuck))
            parts += [f'{r.steps_over_tau_min:.1f}', f'{r.rhat_max:.3f}']
        lines.append(' & '.join(parts) + r' \\')
    return '\n'.join(lines)


def write_tables(table, out_dir):
    table.to_csv(os.path.join(out_dir, 'chain_diagnostics.csv'), index=False, float_format='%.4f')
    with open(os.path.join(out_dir, 'chain_diagnostics_selected.tex'), 'w', encoding='utf-8') as fh:
        fh.write(latex_selected(table) + '\n')
    if table['chain'].str.startswith('loo').any():
        with open(os.path.join(out_dir, 'chain_diagnostics_loo.tex'), 'w', encoding='utf-8') as fh:
            fh.write(latex_loo(table) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=os.path.join('Output_Calibration', 'thesis_figures', 'chains'))
    ap.add_argument('--skip-loo', action='store_true')
    ap.add_argument('--table-only', action='store_true', help='rewrite the LaTeX tables from chain_diagnostics.csv')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.table_only:
        table = pd.read_csv(os.path.join(args.out_dir, 'chain_diagnostics.csv'))
    else:
        rows = []
        selected_figures(args.out_dir, rows)
        if not args.skip_loo:
            loo_figures(args.out_dir, rows)
        table = pd.DataFrame(rows)
    write_tables(table, args.out_dir)
    print(table[['chain', 'steps', 'walkers', 'stuck', 'tau_max', 'steps_over_tau_min', 'rhat_max']].to_string(index=False))


if __name__ == '__main__':
    main()
