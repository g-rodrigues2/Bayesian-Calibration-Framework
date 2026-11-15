"""Sensitivity of the prior predictive check to the upper bound of the decay rate of CD and ACD.

    CALIB_GROWTH_SOURCE=fixed python ppc_bound_sweep.py [--draws 20000] [--seeds 0 1 2] [--workers 6]

Supports the second qualification of the prior predictive check in Chapter 4 of the thesis
(Section on the immunotherapy group, paragraph "Two qualifications apply to this diagnostic").

The protocol is that of ppc_grid.py: one random generator per seed, consumed by the models in the
order of THESIS_MODELS, so that with the adopted bounds the result for seed 0 is the one shown in the
figure. Only the upper bound of beta_immuno (mu in the thesis) of the swept model is changed; every
other bound is that of bounds_imm.py. For each model, bound and seed, the number of observed volumes
outside the 5-95% band of the simulated observations of the same mouse is counted over the three
mice. Writes Output_Calibration/ppc_bound_sweep_io_sensitive.csv.
"""
import _path_setup  # noqa: F401

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import bounds_imm
import ppc_grid
from data_loader_imm import load_immuno_data
from growth_params import load_growth_params

SWEEP = {
    '_CumulativeDose': [8.0, 9.0, 10.0, 11.0, 15.0],
    '_AccelCD': [40.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0, 300.0, 500.0],
}


def outside(model, rng, mice, growth, max_ic, n_draws):
    """Observed volumes outside the 5-95% band of their own mouse, and number of observations."""
    times, sims = ppc_grid.simulate(model, mice, growth, max_ic, n_draws, rng)
    n_out = n_obs = 0
    for mouse, sim in zip(mice, sims):
        lo, hi = np.percentile(sim, [5, 95], axis=0)
        idx = np.searchsorted(times, mouse['data'][:, 0])
        obs = mouse['data'][:, 1]
        n_out += int(np.sum((obs < lo[idx]) | (obs > hi[idx])))
        n_obs += len(obs)
    return n_out, n_obs


def run(cfg):
    model, upper, seed, n_draws = cfg
    labels, low, up = bounds_imm._TREAT_BOUNDS[model]
    bounds_imm._TREAT_BOUNDS[model] = (labels, low, [up[0], upper])
    full = load_immuno_data()
    mice = full[ppc_grid.GROUP]
    growth = load_growth_params([ppc_grid.GROUP], source='fixed')
    max_ic = bounds_imm.find_max_initial_condition(full)
    rng = np.random.default_rng(seed)
    for previous in ppc_grid.THESIS_MODELS[:ppc_grid.THESIS_MODELS.index(model)]:
        outside(previous, rng, mice, growth, max_ic, n_draws)  # keeps the random stream of ppc_grid.py
    n_out, n_obs = outside(model, rng, mice, growth, max_ic, n_draws)
    return dict(model=model, upper_bound=upper, seed=seed, n_outside=n_out, n_obs=n_obs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--draws', type=int, default=20000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default='./Output_Calibration')
    args = ap.parse_args()

    cfgs = [(model, upper, seed, args.draws) for model, uppers in SWEEP.items()
            for upper in uppers for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        table = pd.DataFrame(list(ex.map(run, cfgs)))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'ppc_bound_sweep_{ppc_grid.GROUP}.csv')
    table.to_csv(path, index=False)
    print(table.pivot_table(index=['model', 'upper_bound'], columns='seed', values='n_outside').to_string())
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
