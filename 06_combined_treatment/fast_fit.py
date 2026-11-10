"""Maximum-likelihood fit of a combined-treatment fold in about a second.

The strict leave-one-out comparison of Section 4.7 scores each held-out mouse with the
maximum-likelihood estimate of its training fold, which the MCMC supplies as its best sample. A fold
has 17 parameters, but they separate:

    the residual standard deviation has a closed form, since for a Gaussian likelihood with a common
    sigma the profile maximum is sigma^2 = SSE/n, and the profiled log-likelihood reduces to
    -n/2 [1 + log(2 pi SSE/n)], a monotone function of the total sum of squares;

    each mouse-specific initial volume enters the likelihood only through its own mouse, so the
    fourteen of them are fourteen independent one-dimensional searches rather than fourteen
    dimensions of a joint search.

What remains is a search over the coefficients of the lysis function, two of them at most, which a
coarse grid followed by a bounded Powell refinement covers in a fraction of a second. This is the
same profiling strategy that 07_bic_verification/fast_profile.py applies to the radiotherapy stage.

    python fast_fit.py --models _SUVExp,_VolExp        # write estimate_*.npz for every fold
    python fast_fit.py --check                         # compare with the estimates of the chains
"""
import argparse
import itertools
import os
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import _path_setup  # noqa: F401
import rad_immuno_fast as fast
from bounds_ri import PARAM_BOUNDS, IC_UPPER_FACTOR
from data_loader_ri import load_data, max_initial_volume
from rad_immuno_models import MODEL_PARAMS, lysis_coefficient, standardized_upstream, COVARIATE
from thesis_params import upstream_params

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'Output_Calibration', 'loo')
GROUP = 'rad+io_sensitive'
RTOL, ATOL = 1e-7, 1e-6          # 2.3x faster than the default and still 1e-4 mm3 accurate
GRID = 7                          # coarse points per coefficient before the local refinement
N_STARTS = 5                      # best grid points refined; with one, the maximum depends on the grid
SCAN_STEP = 1.0                   # mm3 between scanned initial volumes (see _profile)
N_REFINE = 3                      # brackets of the scan refined by a local search


def _phis(model, params, cohort, upstream):
    values = cohort.volume0 if COVARIATE.get(model) == 'volume0' else cohort.suv
    return np.array([float(lysis_coefficient(model, params, v, upstream)) for v in values])


def _sse_of_mouse(cohort, m, ic, phi, upstream, grid):
    ev2d, n_ev, nrad2d, nio2d, tr = grid
    n, e = int(cohort.n_obs[m]), int(n_ev[m])
    pred = fast.predict_on_grid(float(ic), float(phi), cohort.times[m], n, upstream['r'],
                                upstream['cc'], upstream['a_rad'], upstream['b_rad'],
                                upstream['delay'], upstream['s'], upstream['lam'], tr, 2.0, 1.0,
                                RTOL, ATOL, ev2d[m, :e], nrad2d[m, :e], nio2d[m, :e])
    if not np.all(np.isfinite(pred)):
        return np.inf
    d = pred - cohort.volumes[m, :n]
    return float(d @ d)


def _profile(model, params, cohort, upstream, grid, ic_hi):
    """Total SSE with every initial volume at its own optimum, and those optima."""
    phis = _phis(model, params, cohort, upstream)
    if not np.all(np.isfinite(phis)):
        return np.inf, None
    total, ics = 0.0, np.empty(len(cohort.names))
    for m in range(len(cohort.names)):
        def sse(ic, m=m):
            return _sse_of_mouse(cohort, m, ic, phis[m], upstream, grid)
        # The sum of squares is far from unimodal in the initial volume. For some mice it has wells
        # only a few mm3 wide, where the tumor switches between control and escape, separated by
        # much flatter regions; a bracket-based local search steps over them. The range is therefore
        # scanned at a resolution that resolves such a well, and the best brackets are refined.
        n_scan = max(32, int(np.ceil(ic_hi / SCAN_STEP)) + 1)
        xs = np.linspace(1e-6, ic_hi, n_scan)
        fs = np.array([sse(x) for x in xs])
        best = None
        for i in sorted(np.argsort(fs)[:N_REFINE]):
            lo_i, hi_i = xs[max(i - 1, 0)], xs[min(i + 1, n_scan - 1)]
            if hi_i <= lo_i:
                continue
            res = minimize_scalar(sse, bounds=(lo_i, hi_i), method='bounded',
                                  options={'xatol': 1e-8, 'maxiter': 200})
            if best is None or res.fun < best.fun:
                best = res
        total += best.fun
        ics[m] = best.x
    return total, ics


def fit(model, mice, upstream, ic_hi):
    """Maximum-likelihood theta = [coefficients, initial volumes, std] of one fold."""
    cohort = fast.cohort(mice)
    grid = cohort.grid(upstream['delay'])
    n_obs = int(cohort.n_obs.sum())
    lo, hi = PARAM_BOUNDS[model]
    k = len(MODEL_PARAMS[model])

    def objective(p):
        sse, _ = _profile(model, tuple(p), cohort, upstream, grid, ic_hi)
        return np.inf if not np.isfinite(sse) else 0.5 * n_obs * (1.0 + np.log(2 * np.pi * sse / n_obs))

    if k == 0:
        best_p = ()
    else:
        axes = [np.linspace(lo[i] + 1e-6, hi[i] - 1e-6, GRID) for i in range(k)]
        scored = sorted(((objective(np.array(p)), p) for p in itertools.product(*axes)),
                        key=lambda t: t[0])
        best_f, best_p = scored[0]
        for _, start in scored[:N_STARTS]:
            res = minimize(objective, np.array(start), method='Powell',
                           bounds=list(zip(lo, hi)), options={'xtol': 1e-6, 'ftol': 1e-8})
            if res.fun < best_f:
                best_f, best_p = float(res.fun), tuple(res.x)

    sse, ics = _profile(model, best_p, cohort, upstream, grid, ic_hi)
    std = float(np.sqrt(sse / n_obs))
    ll = float(-0.5 * (sse / std ** 2 + n_obs * np.log(2 * np.pi * std ** 2)))
    return np.concatenate([np.array(best_p, dtype=float), ics, [std]]), ll


def folds(models, selected=None):
    mice = load_data()[GROUP]
    base = upstream_params('sensitive')
    ic_hi = IC_UPPER_FACTOR * max_initial_volume(mice)
    for model in models:
        for held_out in mice:
            if selected and held_out['name'] not in selected:
                continue
            training = [m for m in mice if m['name'] != held_out['name']]
            yield model, held_out['name'], training, standardized_upstream(model, base, training), ic_hi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default='_SUVExp,_VolExp')
    ap.add_argument('--mice', default=None)
    ap.add_argument('--check', action='store_true',
                    help='compare with estimate_*.npz of the chains instead of writing')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    selected = args.mice.split(',') if args.mice else None

    t0 = time.perf_counter()
    rows = []
    for model, name, training, upstream, ic_hi in folds(args.models.split(','), selected):
        start = time.perf_counter()
        theta, ll = fit(model, training, upstream, ic_hi)
        stem = f'{GROUP}_EXCL-{name}{model}'
        path = os.path.join(OUT, f'estimate_{stem}.npz')
        if args.check:
            if not os.path.exists(path):
                print(f'{stem}: no chain estimate to compare')
                continue
            old = np.load(path)
            rows.append((stem, float(old['max_ll']), ll, time.perf_counter() - start))
            print(f'{stem:52} chain {old["max_ll"]:12.4f}   profile {ll:12.4f}   '
                  f'{ll - float(old["max_ll"]):+8.4f}   {time.perf_counter() - start:5.2f}s')
        else:
            labels = list(MODEL_PARAMS[model]) + [f'ic{i + 1}' for i in range(len(training))] + ['std']
            np.savez(path, max_ll=ll, pars=theta, labels=np.array(labels), n_steps=0,
                     n_stuck_walkers=0)
            print(f'{stem:52} ll {ll:12.4f}   {time.perf_counter() - start:5.2f}s')
    print(f'\ntotal {time.perf_counter() - t0:.1f}s')
    if rows:
        better = sum(1 for _, a, b, _ in rows if b >= a - 1e-6)
        print(f'profile at least as good as the chain in {better} of {len(rows)} folds')


if __name__ == '__main__':
    main()
