"""Sensitivity of the leave-one-out comparison to the summary of the immunotherapy posterior inherited.

The combined-treatment stage inherits (d, s, lam) from the immunotherapy calibration, whose maximum
lysis rate is only weakly identified, so the question is whether the leave-one-out comparison of Chapter 4
depends on which summary of that posterior is carried forward. This repeats the strict comparison,
all five formulations in the same fifteen folds, under the three summaries one could reasonably
inherit:

    max_likelihood   the sample of largest likelihood, the vector of the LOO table (thesis_params.py)
    marginal_median  the marginal medians of the converged immunotherapy chain
    submitted        the vector the earlier version inherited, which traces to the legacy calibration

Each fold is fitted exactly as the LOO table is, by the profiled maximum of fast_fit.py with its
multi-start refinement, and scored as loo.py strict scores it. Under max_likelihood nothing is
refitted: the fold estimates of stage 6 are read and the per-mouse scores are checked against
strict_loo_per_mouse.csv, so this script cannot report a LOO table different from the thesis.

    python vector_sensitivity.py                 # every vector, one process per core
    python vector_sensitivity.py --processes 8
    python vector_sensitivity.py --check-only    # only the max_likelihood check, in seconds

The 150 refits of the two alternative vectors take about 27 CPU-hours.

Outputs: results/vector_sensitivity_per_fold.csv and results/vector_sensitivity_summary.csv. The
summary gives, per vector, the total held-out log-likelihood of each formulation, the count of mice
above M0 and the advantage of the best formulation with SUV over the best without it, which is the
quantity Section 4.7 of the thesis quotes.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

import numpy as np
import pandas as pd

# Every stage has a module of this name. A spawned worker inherits the parent's sys.path, on which the
# stages already come first, so this folder is put in front to import its own.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stage_paths  # noqa: F401,E402  (stages and tools on sys.path)
from _stage_paths import RESULTS, stage

NON_SUV_ARMS = ('_RadImmunoFixed', '_RadImmunoFitD')
MODELS = ['_SUVLinear', '_SUVExp', '_VolExp', '_RadImmunoFixed', '_RadImmunoFitD']   # LOO table order
WITH_SUV = ('_SUVLinear', '_SUVExp')     # _VolExp replaces SUV by the baseline volume
M0 = '_RadImmunoFitD'
RETRIES = 2            # fresh executors for folds lost with a dead worker
# The vector an earlier version inherited. It is not the maximum nor the median of any stored chain of
# the immunotherapy stage and traces to the legacy calibration; it is kept here as a constant because
# it is the summary that was carried forward before, which is the point of comparing against it.
SUBMITTED = dict(d=1.5243135102396321, s=0.015152682754421672, lam=1.928733156897561)
IMMUNO_CHAIN = os.path.join(stage('05_immunotherapy'), 'Output_Calibration',
                            'multi_chain_io_sensitive_dePillisRadunskayaLaw.gz')
IMMUNO_LABELS = ('d', 's', 'lam')        # leading columns of that chain, as in multi_ll_pars_*.npz
STRICT = os.path.join(stage('06_combined_treatment'), 'Output_Calibration', 'loo',
                      'strict_loo_per_mouse.csv')


def vectors():
    from thesis_params import upstream_params
    base = upstream_params('sensitive')
    chain = np.loadtxt(IMMUNO_CHAIN)
    median = {k: float(np.median(chain[:, i])) for i, k in enumerate(IMMUNO_LABELS)}
    return {'max_likelihood': base,
            'marginal_median': {**base, **median},
            'submitted': {**base, **SUBMITTED}}


def score(model, params, std, mouse, up):
    """Held-out log-likelihood of one mouse, as loo.py strict computes it."""
    from inference_ri import gaussian_loglik
    from rad_immuno_models import solve_mouse
    obs = mouse['data'][:, 1]
    pred = solve_mouse(model, params, obs[0], mouse, up)
    return gaussian_loglik(obs[1:], pred[1:], std)


def refit(task):
    """Fit one fold under one vector and score its held-out mouse. Runs in a worker process."""
    vector_name, vector, model, held_out = task
    from bounds_ri import IC_UPPER_FACTOR
    from data_loader_ri import load_data, max_initial_volume
    from fast_fit import fit
    from inference_ri import split_theta
    from loo import GROUP, fold_setup
    mice = load_data()[GROUP]
    mouse = next(m for m in mice if m['name'] == held_out)
    training, up = fold_setup(mice, model, mouse, vector)
    t0 = time.perf_counter()
    theta, train_ll = fit(model, training, up, IC_UPPER_FACTOR * max_initial_volume(mice))
    params, _, std = split_theta(theta, model, 0)
    return dict(vector=vector_name, model=model, mouse=held_out, train_ll=train_ll,
                heldout_ll=score(model, params, std, mouse, up), seconds=time.perf_counter() - t0)


def published(vector):
    """The LOO table itself, from the fold estimates of stage 6."""
    from data_loader_ri import load_data
    from loo import GROUP, fold_estimate, fold_setup
    rows = []
    mice = load_data()[GROUP]
    for model in MODELS:
        for mouse in mice:
            _, up = fold_setup(mice, model, mouse, vector)
            params, std = fold_estimate(model, mouse['name'])
            rows.append(dict(vector='max_likelihood', model=model, mouse=mouse['name'],
                             train_ll=np.nan, heldout_ll=score(model, params, std, mouse, up),
                             seconds=0.0))
    return rows


def summarize(per_fold):
    rows = []
    for vector, df in per_fold.groupby('vector', sort=False):
        ell = df.pivot(index='mouse', columns='model', values='heldout_ll')
        best_suv = max(WITH_SUV, key=lambda m: ell[m].sum())
        best_non = max(NON_SUV_ARMS, key=lambda m: ell[m].sum())
        row = {'vector': vector}
        for m in MODELS:
            row[f'total{m}'] = ell[m].sum()
            if m != M0:
                row[f'above_M0{m}'] = int((ell[m] > ell[M0]).sum())
        order = sorted(MODELS, key=lambda m: -ell[m].sum())
        row.update(order=' > '.join(order), best_suv=best_suv, best_non_suv=best_non,
                   advantage=ell[best_suv].sum() - ell[best_non].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processes', type=int, default=os.cpu_count())
    ap.add_argument('--check-only', action='store_true',
                    help='stop after reproducing the LOO table from the fold estimates (no refits, '
                         'nothing written)')
    args = ap.parse_args()
    from data_loader_ri import load_data
    from loo import GROUP

    vecs = vectors()
    rows = published(vecs['max_likelihood'])
    strict = pd.read_csv(STRICT).set_index(['model', 'mouse'])['heldout_ll']
    worst = max(abs(r['heldout_ll'] - strict[(r['model'], r['mouse'])]) for r in rows)
    if worst > 1e-5:              # the CSV holds six decimals
        raise RuntimeError(f'the fold estimates do not reproduce strict_loo_per_mouse.csv '
                           f'(largest difference {worst:.3g}); rerun loo.py strict first')
    print(f'max_likelihood: LOO table reproduced from the fold estimates (largest difference {worst:.1e})')
    if args.check_only:
        return

    names = [m['name'] for m in load_data()[GROUP]]
    tasks = [(v, vecs[v], model, name) for v in ('marginal_median', 'submitted')
             for model in MODELS for name in names]
    # The formulations with a lysis function to search take minutes per fold and those without take
    # one, so the slow ones go first and the pool does not end waiting on a single fold.
    tasks.sort(key=lambda t: t[2] in NON_SUV_ARMS)
    print(f'{len(tasks)} folds to refit on {args.processes} processes', flush=True)
    t0 = time.time()
    # A worker that dies (on Windows, a busy machine has been seen to kill an interpreter while it
    # imports pandas) breaks the executor instead of hanging it; the folds it did not return are then
    # submitted again to a fresh one.
    pending, done = list(tasks), 0
    for _ in range(1 + RETRIES):
        lost = []
        with ProcessPoolExecutor(args.processes, mp_context=get_context('spawn')) as pool:
            futures = {pool.submit(refit, t): t for t in pending}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except BrokenProcessPool:
                    lost.append(futures[fut])
                    continue
                rows.append(r)
                done += 1
                print(f'  [{done:3d}/{len(tasks)}] {r["vector"]:15} {r["model"]:15} {r["mouse"]:7} '
                      f'held-out {r["heldout_ll"]:10.3f}  {r["seconds"]:6.0f}s  '
                      f'({(time.time() - t0) / 60:.0f} min)', flush=True)
        if not lost:
            break
        print(f'  {len(lost)} folds lost with a worker; submitting them again', flush=True)
        pending = lost
    else:
        raise RuntimeError(f'{len(lost)} folds could not be computed')

    per_fold = pd.DataFrame(rows)
    per_fold['vector'] = pd.Categorical(per_fold['vector'], list(vecs), ordered=True)
    per_fold['model'] = pd.Categorical(per_fold['model'], MODELS, ordered=True)
    per_fold = per_fold.sort_values(['vector', 'model', 'mouse']).reset_index(drop=True)
    summary = summarize(per_fold)

    os.makedirs(RESULTS, exist_ok=True)
    for name, df in (('vector_sensitivity_per_fold.csv', per_fold),
                     ('vector_sensitivity_summary.csv', summary)):
        path = os.path.join(RESULTS, name)
        df.to_csv(path, index=False, float_format='%.6f')
    print(summary.to_string())


if __name__ == '__main__':
    main()
