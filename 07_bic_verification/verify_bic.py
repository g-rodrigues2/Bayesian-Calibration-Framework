"""Verification of the maximum log-likelihoods behind the BIC tables of Chapter 4.

The BIC of each model uses only the maximum log-likelihood, which the calibrations took from the best MCMC
sample. For every cell of the BIC tables this script
  1. re-evaluates the log-likelihood of the stored maximum-likelihood sample with the code of this
     repository (reproduction check), and
  2. maximizes the log-likelihood inside the prior box by bounded Powell searches restarted until no
     further improvement, from the stored sample, from perturbations of it and, for the low-dimensional
     mouse-specific problems, from uniform draws over the box.
The refined maximum can only be equal to or higher than the stored one when the stored sample lies inside
the box, so BIC values can only decrease.

    python verify_bic.py import --legacy <folder of the original calibrations>   # once; fills estimates/
    python verify_bic.py run --scope global --workers 14      # group and scenario calibrations first
    python verify_bic.py run --scope mouse --workers 14       # mouse-specific calibrations
    python verify_bic.py run --tables S2.2 --scope all
    python summarize_bic.py                                   # tables before/after, ranking changes

Results are appended to results/cells.csv (one row per cell; cells already present are skipped, so an
interrupted run resumes) and the refined parameter vectors to results/pars/<cell id>.npz.
"""
import argparse
import csv
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import _stage_paths  # noqa: F401,E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from entries import all_cells, import_estimates, load_estimate  # noqa: E402
from likelihoods import problem, safe  # noqa: E402
import fast_profile  # noqa: E402

HERE = _stage_paths.HERE
RESULTS = os.path.join(HERE, 'results')
# BIC_CELLS_CSV writes to another file, so that two runs (e.g. global and mouse scopes) can go at the same time
CELLS_CSV = os.environ.get('BIC_CELLS_CSV') or os.path.join(RESULTS, 'cells.csv')
FIELDS = ['id', 'table', 'block', 'row', 'column', 'model', 'n', 'k', 'stored_ll', 'reproduced_ll',
          'stored_inside_box', 'refined_ll', 'delta_ll', 'bic_stored', 'bic_refined', 'winning_start',
          'n_starts', 'n_evals', 'seconds', 'near_bound']

# Budgets: (evaluations per Powell search, restarts per start, uniform starts)
BUDGET = {'mouse': (3000, 4, 6), 'global': (20000, 4, 0)}
# Radiotherapy cells maximized through the exact profile of fast_profile: (evaluations, restarts, uniform starts)
PROFILE_BUDGET = (3000, 4, 40)


def bic(ll, k, n):
    return k * np.log(n) - 2.0 * ll


def maximize(f, lb, ub, starts, maxfev, rounds, tol=1e-6):
    """Best (theta, loglik, index of the start, evaluations) of bounded Powell searches from each start.
    Each search is restarted from its own result until it improves the log-likelihood by less than tol."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    eps = 1e-9 * np.maximum(ub - lb, 1e-12)
    lo, hi = lb + eps, ub - eps
    count = [0]

    def neg(x):
        count[0] += 1
        v = f(x)
        return -v if np.isfinite(v) else 1e300

    best = (None, -np.inf, -1)
    for i, s in enumerate(starts):
        x = np.clip(np.asarray(s, float), lo, hi)
        fx = neg(x)
        for _ in range(rounds):
            res = minimize(neg, x, method='Powell', bounds=list(zip(lo, hi)),
                           options=dict(maxfev=maxfev, xtol=1e-8, ftol=1e-12))
            xr = np.clip(res.x, lo, hi)
            fr = neg(xr)
            if fr < fx:
                gain = fx - fr
                x, fx = xr, fr
                if gain < tol:
                    break
            else:
                break
        if -fx > best[1]:
            best = (x, -fx, i)
    return best[0], best[1], best[2], count[0]


def _sibling_estimate(cell, model):
    """Stored estimate of another model in the same table cell position (same block and row)."""
    return load_estimate(dict(cell, model=model, id=cell['id'].replace(cell['model'], model)))


# The delayed onset-factor models are started from the delayed model without the factor. Their parameter
# order is (alpha, beta, delay, rate): the original calibrations of these two models activated the effect
# at the dose day instead of dose day + delay, so their stored samples are poor starts for the model of
# the thesis.
ONSET_FACTOR_BASE = {'_EDEFDelay': '_ExponentialDecayDelay', '_CDEFDelay': '_CumulativeDoseDelay'}


def starting_points(cell, lb, ub, stored, rng):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    width = ub - lb
    starts = []
    if stored is not None:
        starts.append(stored)
        for scale in (0.02, 0.02, 0.1, 0.1):
            spread = np.maximum(np.abs(stored), 1e-3 * width) * scale
            starts.append(stored + spread * rng.normal(size=stored.size))
    elif cell['model'] == '_SUVLogistic':
        linear = _sibling_estimate(cell, '_SUVLinear')
        rest = linear[0][2:] if linear is not None else (lb[2:] + ub[2:]) / 2
        for a in (0.5, 1.5, 2.5):
            for b in (-1.0, 0.5, 2.0):
                starts.append(np.concatenate([[a, b], rest]))
    if cell['model'] in ONSET_FACTOR_BASE:
        base = _sibling_estimate(cell, ONSET_FACTOR_BASE[cell['model']])
        if base is not None:
            alpha, beta, delay, rest = base[0][0], base[0][1], base[0][2], base[0][3:]
            for d in (min(delay, 0.9 * ub[2]), 0.5 * ub[2]):
                for rate in (0.05, 0.25, 0.75):
                    starts.append(np.concatenate([[alpha, beta, d, lb[3] + rate * (ub[3] - lb[3])], rest]))
    scope = 'global' if cell['row'] in ('Global',) or cell['kind'].endswith('sharing') else 'mouse'
    n_uniform = BUDGET[scope][2]
    if scope == 'global' and cell['kind'] == 'immuno':
        n_uniform = 40                   # low-dimensional group problems: explore the box as well
    for _ in range(n_uniform):
        starts.append(lb + width * rng.uniform(size=width.size))
    return starts, scope


def verify_cell(cell):
    t0 = time.time()
    loglik, lb, ub, labels, n = problem(cell)
    f = safe(loglik)
    k = len(lb)
    estimate = load_estimate(cell)
    rng = np.random.default_rng(zlib.crc32(cell['id'].encode()))
    stored_pars, stored_ll = (estimate if estimate is not None else (None, np.nan))
    reproduced = f(stored_pars) if stored_pars is not None else np.nan
    inside = (stored_pars is not None and len(stored_pars) == k
              and bool(np.all((stored_pars > np.asarray(lb)) & (stored_pars < np.asarray(ub)))))
    if stored_pars is not None and len(stored_pars) != k:
        raise ValueError(f'{cell["id"]}: stored vector has {len(stored_pars)} entries, bounds {k}')
    starts, scope = starting_points(cell, lb, ub, stored_pars, rng)
    if fast_profile.covered(cell):
        # maximize over the radiotherapy parameters only; initial volumes and std are maximized exactly
        profile, full, n_r = fast_profile.profile_problem(cell, lb, ub)
        maxfev, rounds, n_uniform = PROFILE_BUDGET
        lo_r, hi_r = np.asarray(lb[:n_r], float), np.asarray(ub[:n_r], float)
        starts = [np.asarray(s[:n_r], float) for s in starts]
        starts += [lo_r + (hi_r - lo_r) * rng.uniform(size=n_r) for _ in range(n_uniform)]
        theta_r, _, winner, evals = maximize(profile, lo_r, hi_r, starts, maxfev, rounds)
        theta = full(theta_r)
        ll = f(theta)                    # reported with the likelihood of the calibrations (odeint)
    else:
        maxfev, rounds, _ = BUDGET[scope]
        theta, ll, winner, evals = maximize(f, lb, ub, starts, maxfev, rounds)
    if inside and reproduced > ll:
        theta, ll, winner = stored_pars, reproduced, 0
    width = np.asarray(ub) - np.asarray(lb)
    near = [lab for lab, v, lo, hi, w in zip(labels, theta, lb, ub, width)
            if min(v - lo, hi - v) < 1e-3 * w]
    os.makedirs(os.path.join(RESULTS, 'pars'), exist_ok=True)
    np.savez(os.path.join(RESULTS, 'pars', cell['id'] + '.npz'), pars=theta, max_ll=ll, labels=np.array(labels))
    return dict(id=cell['id'], table=cell['table'], block=cell['block'], row=cell['row'], column=cell['column'],
                model=cell['model'], n=n, k=k, stored_ll=stored_ll, reproduced_ll=reproduced,
                stored_inside_box=inside, refined_ll=ll,
                delta_ll=ll - reproduced if np.isfinite(reproduced) else np.nan,
                bic_stored=bic(stored_ll, k, n) if np.isfinite(stored_ll) else np.nan,
                bic_refined=bic(ll, k, n), winning_start=winner, n_starts=len(starts), n_evals=evals,
                seconds=round(time.time() - t0, 1), near_bound=';'.join(near))


def _done():
    if not os.path.exists(CELLS_CSV):
        return set()
    with open(CELLS_CSV, newline='') as fh:
        return {row['id'] for row in csv.DictReader(fh)}


def run(cells, workers):
    os.makedirs(RESULTS, exist_ok=True)
    done = _done()
    todo = [c for c in cells if c['id'] not in done]
    # expensive problems first, so that long searches start while cheap ones fill the remaining workers
    todo.sort(key=lambda c: (c['row'] != 'Global' and not c['kind'].endswith('sharing'), c['id']))
    print(f'{len(todo)} cells to verify ({len(cells) - len(todo)} already done)', flush=True)
    new = not os.path.exists(CELLS_CSV)
    with open(CELLS_CSV, 'a', newline='') as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        futures = {pool.submit(verify_cell, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            cell = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                print(f'[{i}/{len(todo)}] {cell["id"]} FAILED: {exc!r}', flush=True)
                continue
            writer.writerow(row)
            fh.flush()
            print(f'[{i}/{len(todo)}] {row["id"]} stored {row["stored_ll"]:.3f} reproduced '
                  f'{row["reproduced_ll"]:.3f} refined {row["refined_ll"]:.3f} ({row["seconds"]:.0f} s)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['import', 'run'])
    ap.add_argument('--legacy', default=None, help='folder of the original calibrations (import)')
    ap.add_argument('--tables', default=None, help='comma-separated table labels (default: all)')
    ap.add_argument('--scope', choices=['global', 'mouse', 'all'], default='all')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    cells = all_cells()
    if args.tables:
        wanted = set(args.tables.split(','))
        cells = [c for c in cells if c['table'] in wanted]
    if args.scope != 'all':
        is_global = lambda c: c['row'] == 'Global' or c['kind'].endswith('sharing')  # noqa: E731
        cells = [c for c in cells if is_global(c) == (args.scope == 'global')]

    if args.action == 'import':
        missing = import_estimates(cells, args.legacy)
        print(f'{len(cells) - len(missing)} estimates imported; missing: {len(missing)}')
        for m in missing:
            print('  ', m)
    else:
        run(cells, args.workers)


if __name__ == '__main__':
    main()
