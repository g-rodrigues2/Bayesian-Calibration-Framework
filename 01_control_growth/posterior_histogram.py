"""Posterior histograms of the calibrated parameters, with the maximum-likelihood sample marked.

For each model parameter (initial conditions and std are excluded):
  - posterior density histogram
  - dashed red line at the maximum-likelihood sample (from the .npz)
  - solid line at the median and dotted lines at the 95% credible interval

Works with the chains of both modes:
  global          -> multi_chain_{group}{ext}.gz  (labels: r, cc, ic1..icM, std)
  mouse_specific  -> chain_{group}_{mouse}{ext}.gz (labels: r, cc, ic, std)

Labels are read from the .npz written by global.py; if absent, they are rebuilt with
bounds.define_bounds_labels.

    from posterior_histogram import plot_posterior_histograms
    plot_posterior_histograms(
        chain_path='Output_Calibration/multi_chain_control_resistant_log.gz',
        npz_path='Output_Calibration/multi_ll_pars_control_resistant_log.npz',
        figure_basename='Output_Calibration/posterior_hist_control_resistant_log',
    )

    python posterior_histogram.py \
        --chain Output_Calibration/multi_chain_control_resistant_log.gz \
        --npz   Output_Calibration/multi_ll_pars_control_resistant_log.npz \
        --out   Output_Calibration/posterior_hist_control_resistant_log
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from postprocess import is_model_param


def labels_for(model_extension, group_size, max_ic, bool_multiple_mice):
    """Labels rebuilt from bounds.py (fallback when the .npz has none)."""
    from bounds import define_bounds_labels
    _l, _u, labels = define_bounds_labels(
        model_extension, group_size=group_size, max_ic=max_ic,
        bool_multiple_mice=bool_multiple_mice)
    return labels


def _labels_from_npz(npz_path, n_dim):
    if npz_path is None or not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path, allow_pickle=True)
    if 'labels' not in npz:
        return None
    labels = [str(x) for x in npz['labels']]
    return labels if len(labels) == n_dim else None


def plot_posterior_histograms(chain_path, labels=None, npz_path=None,
                              figure_basename='posterior_hist',
                              bins=50, save=True, show=False):
    """One histogram (one figure) per model parameter.

    labels=None -> read from the .npz. Returns the list of .png paths.
    """
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    n_dim = chain.shape[1]

    if labels is None:
        labels = _labels_from_npz(npz_path, n_dim)
    if labels is None:
        raise ValueError(
            f'Could not determine the labels of {chain_path}. '
            f'Pass labels= explicitly or an .npz that contains them.')
    if len(labels) != n_dim:
        raise ValueError(f'Chain has {n_dim} dimensions; labels has {len(labels)}')

    map_pars = None
    if npz_path is not None and os.path.exists(npz_path):
        map_pars = np.load(npz_path, allow_pickle=True)['pars']

    model_idx = [i for i, lbl in enumerate(labels) if is_model_param(lbl)]
    if not model_idx:
        print(f'[posterior_hist] nothing to plot in {chain_path} (only initial conditions/std)')
        return []

    os.makedirs(os.path.dirname(figure_basename) or '.', exist_ok=True)
    paths = []

    for i in model_idx:
        samples = chain[:, i]
        med = np.median(samples)
        q025, q975 = np.percentile(samples, [2.5, 97.5])
        map_val = float(map_pars[i]) if map_pars is not None else None

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(samples, bins=bins, color='steelblue', alpha=0.55,
                edgecolor='white', density=True)

        ax.axvline(q025, color='gray', linestyle=':', linewidth=1.0,
                   label=f'95% CrI [{q025:.4g}, {q975:.4g}]')
        ax.axvline(q975, color='gray', linestyle=':', linewidth=1.0)
        ax.axvline(med, color='black', linestyle='-', linewidth=1.2,
                   alpha=0.6, label=f'median = {med:.4g}')
        if map_val is not None:
            ax.axvline(map_val, color='crimson', linestyle='--', linewidth=1.8,
                       label=f'MAP = {map_val:.4g}')

        ax.set_xlabel(labels[i], fontsize=13)
        ax.set_ylabel('density', fontsize=13)
        ax.set_title(f'Posterior of {labels[i]}', fontsize=13)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

        if save:
            label_safe = labels[i].replace('/', '_').replace(' ', '_')
            path_png = f'{figure_basename}_{label_safe}.png'
            plt.savefig(path_png, dpi=200, bbox_inches='tight', pad_inches=0.02)
            plt.savefig(f'{figure_basename}_{label_safe}.pdf',
                        bbox_inches='tight', pad_inches=0.02)
            paths.append(path_png)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return paths


def main():
    ap = argparse.ArgumentParser(
        description='Posterior histograms of the model parameters.')
    ap.add_argument('--chain', required=True, help='path of the chain .gz file')
    ap.add_argument('--npz', default=None,
                    help='path of the .npz file (maximum-likelihood sample and labels)')
    ap.add_argument('--out', default=None,
                    help='figure prefix (default: derived from the chain name)')
    ap.add_argument('--labels', default=None,
                    help='comma-separated labels (override the .npz)')
    ap.add_argument('--bins', type=int, default=50)
    args = ap.parse_args()

    out = args.out
    if out is None:
        base = os.path.basename(args.chain).rsplit('.', 1)[0]
        base = base.replace('multi_chain_', '').replace('chain_', '')
        out = os.path.join(os.path.dirname(args.chain) or '.',
                           f'posterior_hist_{base}')

    labels = args.labels.split(',') if args.labels else None
    paths = plot_posterior_histograms(args.chain, labels=labels,
                                      npz_path=args.npz, figure_basename=out,
                                      bins=args.bins)
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
