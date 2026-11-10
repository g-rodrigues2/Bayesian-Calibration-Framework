"""Agreement between the compiled integrator and odeint on posterior samples of the combined cohort.

The MCMC of this stage evaluates the likelihood with the compiled Dormand-Prince 5(4) integrator of
rad_immuno_fast; the original calibrations used scipy's odeint (LSODA). This script draws posterior
samples from the two global chains (SUV-based lysis, _SUVLinear, and the prediction with fixed
parameters, _RadImmunoFixed), evaluates the log-likelihood of each sample with both integrators
(inference_ri.SOLVER = 'compiled' and 'odeint') and writes the absolute differences.

    python solver_agreement.py [--draws 200] [--seed 42]

Writes Output_Calibration/solver_agreement.csv (per formulation: number of draws, seed, maximum and
median absolute difference of the log-likelihood).
"""
import argparse
import os

import numpy as np

import _path_setup  # noqa: F401
import inference_ri
from data_loader_ri import load_data
from thesis_params import upstream_params

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = 'rad+io_sensitive'
OUT = os.path.join(HERE, 'Output_Calibration')
CHAINS = {'_SUVLinear': os.path.join(OUT, 'suv', f'chain_{GROUP}_SUVLinear.gz'),
          '_RadImmunoFixed': os.path.join(OUT, 'prediction', f'chain_{GROUP}_RadImmunoFixed.gz')}


def loglik(theta, model, mice, up, solver):
    inference_ri.SOLVER = solver
    return inference_ri.log_likelihood(theta, model, mice, up)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    mice = load_data()[GROUP]
    up = upstream_params('sensitive')
    rows = []
    for model, chain_path in CHAINS.items():
        labels = np.load(chain_path.replace('chain_', 'estimate_').replace('.gz', '.npz'),
                         allow_pickle=True)['labels']
        chain = np.loadtxt(chain_path, ndmin=2)[:, :len(labels)]
        idx = np.random.default_rng(args.seed).choice(len(chain), size=min(args.draws, len(chain)), replace=False)
        diff = []
        for theta in chain[idx]:
            a = loglik(theta, model, mice, up, 'compiled')
            b = loglik(theta, model, mice, up, 'odeint')
            diff.append(abs(a - b))
        diff = np.array(diff)
        rows.append((model, len(idx), args.seed, float(np.max(diff)), float(np.median(diff))))
        print(f'{model}: {len(idx)} draws, |dLL| max {np.max(diff):.3e}, median {np.median(diff):.3e}')

    path = os.path.join(OUT, 'solver_agreement.csv')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('model,draws,seed,max_abs_diff_ll,median_abs_diff_ll\n')
        for r in rows:
            fh.write('%s,%d,%d,%.6e,%.6e\n' % r)
    print('written:', path)


if __name__ == '__main__':
    main()
