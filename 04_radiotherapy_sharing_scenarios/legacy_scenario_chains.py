"""Converts the chains of the original sharing-scenario calibrations into the format of this stage.

The original runs stored, for each scenario, one flattened post-burn-in chain per phenotype
(multi_chain_radiation_{sensitive,resistant}_DamageRepairTimeDelayV{n}.gz, columns alpha, beta, delay,
initial volumes of the phenotype and std) and did not keep the log-probability of the samples. This
script rebuilds the joint parameter vector of each sample (layout of bounds_scen_rad.py), recovers the
walkers from the row order (steps x walkers), recomputes the log-likelihood of each walker on evenly
spaced steps with the likelihood of this stage, removes the stuck walkers with the rule used in all
calibrations (mean log-likelihood more than 50 units below that of the median walker), and writes
multi_chain__ExponentialDecayDelay_V{n}.gz and multi_ll_pars__ExponentialDecayDelay_V{n}.npz.
Before converting, the maximum-likelihood sample saved with each original run is evaluated with this
likelihood and compared with the stored value, which checks the column mapping.

    CALIB_GROWTH_SOURCE=fixed CALIB_DOSE_DAYS=0,1,2,3,4,5 python legacy_scenario_chains.py \
        --legacy <folder with V1..V8 subfolders> --scenarios 1,3,4,5,6,7,8 --out Output_Calibration/legacy
"""
import argparse
import os

import numpy as np
import pandas as pd

import _path_setup  # noqa: F401
from growth_params import load_growth_params

from bounds_rad import find_max_initial_condition
from bounds_scen_rad import define_scenario_bounds_labels
from data_loader_rad import load_radiation_data, parse_dose_days_env
from inference_scen_rad import log_likelihood_shared
from scenarios_rad import MODEL_PARAM_NAMES, SCENARIOS, WINNER_MODEL

GROUPS = ['radiation_sensitive', 'radiation_resistant']
LEGACY = 'DamageRepairTimeDelay'


def read_chain(path):
    return pd.read_csv(path, sep=r'\s+', header=None, dtype=float).to_numpy()


def joint_columns(labels, n_rad):
    """For each joint label: (phenotype, column) in the original per-phenotype chains."""
    names = MODEL_PARAM_NAMES[WINNER_MODEL]
    cols = []
    for lab in labels:
        if lab == 'std':
            cols.append(('sensitive', -1))
        elif lab.startswith('ic_'):
            _, tag, k = lab.split('_')
            cols.append((tag, n_rad + int(k) - 1))
        else:
            base, tag = lab.rsplit('_', 1)
            cols.append(('sensitive' if tag == 'global' else tag, names.index(base)))
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--legacy', required=True)
    ap.add_argument('--scenarios', default='1,3,4,5,6,7,8')
    ap.add_argument('--walker-steps', type=int, default=20, help='steps per walker used to score walkers')
    ap.add_argument('--tol', type=float, default=50.0)
    ap.add_argument('--target-samples', type=int, default=10000)
    ap.add_argument('--out', default='Output_Calibration/legacy')
    args = ap.parse_args()

    full_data = load_radiation_data(data_dir=_path_setup.DATA_DIR,
                                    dose_days=parse_dose_days_env(os.environ.get('CALIB_DOSE_DAYS')))
    max_ic = find_max_initial_condition(full_data)
    growth = load_growth_params(GROUPS)
    n_rad = len(MODEL_PARAM_NAMES[WINNER_MODEL])
    os.makedirs(args.out, exist_ok=True)

    for n in [int(s) for s in args.scenarios.split(',')]:
        sdef = SCENARIOS[WINNER_MODEL][n]
        _, _, labels = define_scenario_bounds_labels(WINNER_MODEL, sdef, GROUPS, full_data, max_ic)
        cols = joint_columns(labels, n_rad)
        folder = os.path.join(args.legacy, f'V{n}')
        chains, best = {}, {}
        for tag in ('sensitive', 'resistant'):
            chains[tag] = read_chain(os.path.join(folder, f'multi_chain_radiation_{tag}_{LEGACY}V{n}.gz'))
            best[tag] = np.load(os.path.join(folder, f'multi_ll_pars_radiation_{tag}_{LEGACY}V{n}.npz'))
        assert np.allclose(chains['sensitive'][:, -1], chains['resistant'][:, -1]), 'phenotype chains not aligned'
        joint = np.column_stack([chains[tag][:, c] for tag, c in cols])
        theta_ml = np.array([best[tag]['pars'][c] for tag, c in cols])
        ll_ml = log_likelihood_shared(theta_ml, WINNER_MODEL, full_data, GROUPS, sdef, growth)
        stored = float(best['sensitive']['max_ll'])
        print(f'V{n}: stored max_ll {stored:.4f}, recomputed {ll_ml:.4f}')
        if abs(ll_ml - stored) > 1e-3 * max(1.0, abs(stored)):
            raise SystemExit(f'V{n}: column mapping does not reproduce the stored maximum log-likelihood')

        ndim = len(labels)
        candidates = [w for w in (k * ndim for k in range(1, 9)) if joint.shape[0] % w == 0]
        nwalkers = 2 * ndim if 2 * ndim in candidates else candidates[0]   # the original runs used 2 walkers per parameter
        steps = joint.shape[0] // nwalkers
        cube = joint.reshape(steps, nwalkers, ndim)
        idx = np.linspace(0, steps - 1, args.walker_steps).astype(int)
        mean_ll = np.array([np.mean([log_likelihood_shared(cube[s, w], WINNER_MODEL, full_data, GROUPS, sdef, growth)
                                     for s in idx]) for w in range(nwalkers)])
        stuck = mean_ll < np.median(mean_ll) - args.tol
        kept = cube[:, ~stuck, :]
        thin = max(1, kept.shape[0] * kept.shape[1] // args.target_samples)
        flat = kept[::thin].reshape(-1, ndim)
        stem = f'{WINNER_MODEL}_V{n}'
        np.savetxt(os.path.join(args.out, f'multi_chain_{stem}.gz'), flat)
        np.savez(os.path.join(args.out, f'multi_ll_pars_{stem}.npz'), max_ll=ll_ml, pars=theta_ml,
                 labels=np.array(labels), n_stuck=int(stuck.sum()), n_walkers=nwalkers, n_steps_post_burn=steps)
        print(f'V{n}: {nwalkers} walkers x {steps} post-burn-in steps, {stuck.sum()} stuck walkers removed, '
              f'{flat.shape[0]} samples written')


if __name__ == '__main__':
    main()
