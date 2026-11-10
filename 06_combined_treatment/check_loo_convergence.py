"""Convergence of the leave-one-out folds trained here, against the criteria the thesis states.

The numerical-implementation subsection of Chapter 4 commits to two criteria for every reported
posterior: a chain at least fifty integrated autocorrelation times long for every parameter after
the burn-in of one quarter, and a split R-hat of at most 1.05, the same convention as the appendix
tables of 01_control_growth/thesis_chain_traces.py. A chain produced after that sentence was written has to meet them too,
or the sentence stops being true. This reports both, per fold.

The chains are read from the resumable HDF5 backends of reinitiate_files/ when they are present, and
otherwise from the thinned chains of chains/06_combined_treatment/ (export_chains.py), which are the
ones distributed; lengths and autocorrelation times are then reported in steps of the original chain.

Run it only after training has finished. Opening a backend with h5py while its sampler is writing
takes the sampler down on Windows.

    python check_loo_convergence.py --models _SUVLinear,_SUVExp,_VolExp,_RadImmunoFixed,_RadImmunoFitD \
        --out Output_Calibration/loo/convergence_loo.csv
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tnbc/
sys.path.insert(0, os.path.join(ROOT, '01_control_growth'))
sys.path.insert(0, os.path.join(ROOT, '06_combined_treatment'))
sys.path.insert(0, ROOT)

import convergence_check  # noqa: E402,F401  (sets the HDF5 locking env var)
from export_chains import read_exported_chain  # noqa: E402
from walkers import drop_stuck  # noqa: E402

REINIT = os.path.join(ROOT, '06_combined_treatment', 'reinitiate_files')
EXPORTED = os.path.join(ROOT, 'chains', '06_combined_treatment')
TAU_FACTOR = 50.0
RHAT_MAX = 1.05
BUSY_SECONDS = 300


def split_rhat(chain):
    """Split R-hat over walkers, for chain of shape (steps, walkers, ndim).

    Each walker's post-burn-in trace is cut in half and the halves are treated as separate sequences,
    which is what makes the diagnostic sensitive to a walker that has not forgotten where it started.
    """
    n, w, d = chain.shape
    half = n // 2
    seqs = np.concatenate([chain[:half], chain[half:2 * half]], axis=1)   # (half, 2w, ndim)
    m = seqs.shape[1]
    means = seqs.mean(axis=0)                      # (2w, ndim)
    varis = seqs.var(axis=0, ddof=1)               # (2w, ndim)
    w_var = varis.mean(axis=0)
    b_var = half * means.var(axis=0, ddof=1)
    var_hat = (half - 1) / half * w_var + b_var / half
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(var_hat / w_var)


def fold_files(kind, models):
    """Backends of reinitiate_files/ when the calibration is at hand, otherwise the exported chains."""
    for folder, ext in ((REINIT, '.h5'), (EXPORTED, '.npz')):
        if os.path.isdir(folder):
            found = sorted(p for p in os.listdir(folder)
                           if f'{kind}-' in p and any(p.endswith(f'{m}{ext}') for m in models))
            if found:
                return folder, ext, found
    return None, None, []


def post_burn_in(path):
    """(post-burn-in chain without stuck walkers, steps of the original chain, stuck mask, thinning).

    The diagnostics are those of the chain the pipeline keeps: run_chain discards the first quarter
    and drops the walkers stuck in a low-probability region before it writes the draws that every
    later score consumes. A stuck walker left in would report a mixing problem of a chain that is
    never used, and hide nothing about the one that is. An exported chain keeps every `thin`-th step
    and the stuck mask computed on the full chain."""
    if path.endswith('.h5'):
        import emcee
        reader = emcee.backends.HDFBackend(path, read_only=True)
        n = reader.iteration
        kept, _, stuck = drop_stuck(reader.get_chain(discard=n // 4), reader.get_log_prob(discard=n // 4))
        return kept, n, stuck, 1
    chain, _, info = read_exported_chain(path)
    thin, n, stuck = info['thin'], info['n_steps'], np.asarray(info['stuck'], bool)
    first = -(-(n // 4) // thin)                   # first stored step at or after the burn-in
    return chain[first:, ~stuck, :].astype(float), n, stuck, thin


def main():
    import emcee
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default='_RadImmunoFitD,_SUVExp,_VolExp')
    ap.add_argument('--kind', default='EXCL', choices=['EXCL', 'PRED'],
                    help='training folds (EXCL) or the conditional held-out chains of Fig. 14 (PRED)')
    ap.add_argument('--out', default=None,
                    help='also write the table to this CSV (the thesis quotes it)')
    args = ap.parse_args()
    models = args.models.split(',')
    folder, ext, paths = fold_files(args.kind, models)
    if not paths:
        sys.exit(f'no backends of {models} in {REINIT} and no exported chains in {EXPORTED}')
    print(f'reading {len(paths)} chains from {os.path.relpath(folder, ROOT)}\n')

    # A backend written to in the last few minutes belongs to a sampler that is still running, and
    # opening it would take that sampler down.
    busy = [p for p in paths
            if ext == '.h5' and time.time() - os.path.getmtime(os.path.join(folder, p)) < BUSY_SECONDS]
    if busy:
        print(f'skipping {len(busy)} backends still being written: {", ".join(busy)}\n')
        paths = [p for p in paths if p not in busy]

    print(f'{"fold":26} {"steps":>7} {"stuck":>5} {"tau max":>8} {"post/tau":>10} {"R-hat max":>10}  verdict')
    print('-' * 84)
    bad, rows = 0, []
    for name in paths:
        fold = name.split(f'{args.kind}-')[1].replace(ext, '')
        kept, n, stuck, thin = post_burn_in(os.path.join(folder, name))
        rhat = float(np.nanmax(split_rhat(kept)))
        tau = float(np.nanmax(emcee.autocorr.integrated_time(kept, tol=0))) * thin
        post = n - n // 4                          # post-burn-in length, as in thesis_chain_traces.py
        ratio = post / tau if tau == tau and tau > 0 else float('nan')

        ok = (ratio >= TAU_FACTOR) and (rhat <= RHAT_MAX)
        bad += not ok
        print(f'{fold:26} {n:7d} {int(stuck.sum()):5d} {tau:8.1f} {ratio:10.1f} {rhat:10.4f}  '
              f'{"ok" if ok else "FAILS"}')
        rows.append((fold, n, int(stuck.sum()), tau, ratio, rhat, ok))

    if args.out:
        if busy:
            sys.exit('not writing a table that leaves out backends still being written')
        with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('fold,steps,stuck_walkers,tau_max,post_burn_steps_per_tau,split_rhat_max,meets_both\n')
            for r in rows:
                fh.write('%s,%d,%d,%.2f,%.2f,%.5f,%s\n' % r)

    print(f'\n{len(paths) - bad} of {len(paths)} folds meet both criteria '
          f'(post-burn-in steps >= {TAU_FACTOR:.0f} tau and split R-hat <= {RHAT_MAX})')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
