"""Convergence diagnostics of the emcee chains.

For each (group, mouse, model):
  - tau (integrated autocorrelation time); flagged if tau > N/50
  - ESS per walker, N_kept / tau
  - acceptance fraction (0.2-0.5 expected)
  - maximum, median and lower tail of the post-burn-in log-likelihood (stuck walkers)

Backends of both calibration modes are found by glob:
  global mode         -> reinitiate_{group}{model}.h5
  mouse-specific mode -> reinitiate_{group}_{mouse}{model}.h5
The `mouse` column is '-' in the global mode.

    python convergence_check.py
    python convergence_check.py --models _log --groups control_resistant

    from convergence_check import run_convergence_check
    df = run_convergence_check(results_dir='.', save_csv=True, print_table=True)
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from typing import Iterable, Optional

# Must be set before importing emcee (which loads h5py and initializes HDF5). Otherwise reading a
# backend while a calibration writes to it stops the writer with PermissionError (Windows) or EAGAIN (NFS).
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import numpy as np
import pandas as pd
import emcee

from walkers import find_stuck_walkers, ll_tail


DEFAULT_MODELS = ['_exp', '_mendel', '_bert', '_lin', '_gomp', '_surf', '_log']
DEFAULT_GROUPS = ['control_sensitive', 'control_resistant']


def _severity(flags: str) -> int:
    return 0 if flags == 'ok' else len(flags.split(','))


def _flag_chain(tau_max: float, acc_mean: float, ess: float, ll_tail_v: float,
                n_iter: int, n_stuck: int = 0) -> list:
    """Applies the quality thresholds and returns the list of flags.

    `stuck-walkers` uses the lower tail (ll_med - p05), not ll_max - ll_med: stuck walkers lie below
    the median (see walkers.py).
    """
    flags = []
    if tau_max > n_iter / 50:
        flags.append('tau-high')
    if acc_mean < 0.1:
        flags.append('acceptance-low')
    elif acc_mean > 0.7:
        flags.append('acceptance-high')
    if ess < 100:
        flags.append('ESS-low')
    if (np.isfinite(ll_tail_v) and ll_tail_v > 50) or n_stuck > 0:
        flags.append('stuck-walkers')
    return flags


def _analyze_chain(path: str) -> Optional[dict]:
    """Metrics of one HDF5 backend, or None if it is empty."""
    r = emcee.backends.HDFBackend(path, read_only=True)
    n_iter = int(r.iteration)
    if n_iter == 0:
        return None
    nw = int(r.shape[0])
    nd = int(r.shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            tau = r.get_autocorr_time(tol=0)
            tau_max = float(np.nanmax(tau))
            tau_med = float(np.nanmedian(tau))
        except Exception:
            tau_max = float('inf')
            tau_med = float('inf')

    try:
        acc_mean = float(np.mean(r.accepted / r.iteration))
    except Exception:
        acc_mean = float('nan')

    discard = n_iter // 4
    lp2d = r.get_log_prob(discard=discard)          # (iters, nwalkers)
    lp = lp2d.reshape(-1)
    ll_max = float(np.nanmax(lp))
    ll_med = float(np.nanmedian(lp))
    tail = ll_tail(lp)
    n_stuck = int(find_stuck_walkers(lp2d).sum())

    kept = n_iter - discard
    ess = kept / tau_max if tau_max > 0 else float('inf')

    flags = _flag_chain(tau_max, acc_mean, ess, tail, n_iter, n_stuck)
    return dict(
        n_iter=n_iter, ndim=nd, nwalkers=nw,
        tau_max=tau_max, tau_med=tau_med, ESS=ess,
        acceptance=acc_mean, ll_max=ll_max, ll_med=ll_med,
        ll_spread=ll_max - ll_med, ll_tail=tail, n_stuck=n_stuck,
        flags=','.join(flags) if flags else 'ok',
    )


def _discover(reinit_dir: str, group: str, model: str):
    """[(path, mouse_tag)] of the backends of (group, model).

    mouse_tag = '-' in the global mode and the mouse name (e.g. 'c0_m1') in the mouse-specific mode.
    Files with the TEST_ prefix are ignored.
    """
    pattern = os.path.join(reinit_dir, f'reinitiate_{group}*{model}.h5')
    out = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        if base.startswith('reinitiate_TEST_'):
            continue
        middle = base[len(f'reinitiate_{group}'):-len(f'{model}.h5')]
        out.append((path, middle.lstrip('_') or '-'))
    return out


def run_convergence_check(
    results_dir: str = '.',
    models: Iterable = DEFAULT_MODELS,
    groups: Iterable = DEFAULT_GROUPS,
    save_csv: bool = True,
    print_table: bool = True,
) -> pd.DataFrame:
    """Diagnostics of every backend of the given (group, model) pairs.

    Returns a DataFrame with one row per chain (with flags), sorted by severity; with save_csv=True it is
    written to {results_dir}/Output_Calibration/convergence_report.csv.
    """
    reinit = os.path.join(results_dir, 'reinitiate_files')
    rows = []
    for group in groups:
        for model in models:
            for path, mouse in _discover(reinit, group, model):
                metrics = _analyze_chain(path)
                if metrics is None:
                    continue
                metrics['group'] = group
                metrics['mouse'] = mouse
                metrics['model'] = model
                rows.append(metrics)

    if not rows:
        if print_table:
            print(f'[convergence_check] no .h5 backend found in {reinit}')
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['severity'] = df['flags'].apply(_severity)
    front = ['group', 'mouse', 'model', 'ndim', 'nwalkers', 'n_iter', 'n_stuck']
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    if print_table:
        for group in df['group'].unique():
            print(f'\n=== {group} ===')
            sub = df[df['group'] == group].sort_values(
                ['severity', 'model', 'mouse'], ascending=[False, True, True])
            cols = ['mouse', 'model', 'ndim', 'tau_max', 'ESS', 'acceptance',
                    'll_max', 'll_med', 'll_tail', 'n_stuck', 'flags']
            print(sub[cols].to_string(index=False, formatters={
                'tau_max':    lambda x: f'{x:7.1f}' if np.isfinite(x) else '    inf',
                'ESS':        lambda x: f'{x:7.1f}',
                'acceptance': lambda x: f'{x:.3f}',
                'll_max':     lambda x: f'{x:9.2f}',
                'll_med':     lambda x: f'{x:9.2f}',
                'll_tail':    lambda x: f'{x:8.2f}',
                'n_stuck':    lambda x: f'{x:>7d}',
                'flags':      lambda x: x,
            }))

        problematic = df[df['flags'] != 'ok']
        if len(problematic):
            print(f'\n[convergence_check] {len(problematic)} chain(s) with quality '
                  f'flags (see column `flags`).')

    if save_csv:
        out_dir = os.path.join(results_dir, 'Output_Calibration')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'convergence_report.csv')
        df.to_csv(out_path, index=False, float_format='%.3f')
        if print_table:
            print(f'\nCSV: {out_path}')

    return df


def _cli():
    ap = argparse.ArgumentParser(description='Convergence diagnostics of the MCMC chains.')
    ap.add_argument('results_dir', nargs='?', default='.',
                    help='Stage folder (containing reinitiate_files/). Default: cwd.')
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS),
                    help='Comma-separated models.')
    ap.add_argument('--groups', default=','.join(DEFAULT_GROUPS),
                    help='Comma-separated groups.')
    ap.add_argument('--no-csv', action='store_true',
                    help='Do not write convergence_report.csv.')
    args = ap.parse_args()
    run_convergence_check(
        results_dir=args.results_dir,
        models=args.models.split(','),
        groups=args.groups.split(','),
        save_csv=not args.no_csv,
        print_table=True,
    )


if __name__ == '__main__':
    _cli()
