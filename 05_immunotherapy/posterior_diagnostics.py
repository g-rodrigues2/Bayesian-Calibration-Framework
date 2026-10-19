"""Convergence and identifiability diagnostics of an emcee chain (Section 4.4 of the thesis).

    python posterior_diagnostics.py --backend reinitiate_files/reinitiate_io_sensitive_dePillisRadunskayaLaw.h5
    python posterior_diagnostics.py --chain multi_chain_io_sensitive_dePillisRadunskayaLaw.gz --walkers 14 \
        --labels d,s,lam,ic1,ic2,ic3,std --lower 0.01,0.01,0.01,60,60,60,0

For an HDF5 backend the first quarter of the steps is discarded. A flattened chain saved by
emcee.get_chain(flat=True) (step-major order, no thinning) is reshaped with --walkers.

Per parameter: integrated autocorrelation time tau, effective sample size (steps x walkers / tau),
split R-hat computed with the walkers as chains (for an ensemble sampler walkers are not independent
chains, so R-hat is indicative), posterior median and 95% interval, the posterior mass within 5% of
the prior range above the lower bound, and the posterior correlation matrix of the model parameters.
"""
import argparse
import os

import emcee
import numpy as np
import pandas as pd


def split_rhat(x):
    """Split R-hat of x with shape (steps, walkers)."""
    half = x.shape[0] // 2
    chains = np.concatenate([x[:half], x[half:2 * half]], axis=1)
    n = chains.shape[0]
    within = chains.var(axis=0, ddof=1).mean()
    between = n * chains.mean(axis=0).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend')
    ap.add_argument('--chain')
    ap.add_argument('--walkers', type=int)
    ap.add_argument('--labels', required=True, help='comma-separated parameter labels')
    ap.add_argument('--lower', default=None, help='comma-separated prior lower bounds')
    ap.add_argument('--upper', default=None, help='comma-separated prior upper bounds')
    ap.add_argument('--n-model-params', type=int, default=3)
    ap.add_argument('--out', default='./Output_Calibration/posterior_diagnostics.csv')
    args = ap.parse_args()

    if args.backend:
        full = emcee.backends.HDFBackend(args.backend, read_only=True).get_chain()
        x = full[len(full) // 4:]
    else:
        flat = np.loadtxt(args.chain, ndmin=2)
        x = flat.reshape(-1, args.walkers, flat.shape[1])
    labels = args.labels.split(',')
    steps, walkers, ndim = x.shape
    tau = emcee.autocorr.integrated_time(x, quiet=True)
    lower = np.array(args.lower.split(','), float) if args.lower else None
    upper = np.array(args.upper.split(','), float) if args.upper else None

    rows = []
    for i, name in enumerate(labels):
        v = x[:, :, i].ravel()
        lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
        row = dict(parameter=name, tau=tau[i], ess=steps * walkers / tau[i], steps_over_tau=steps / tau[i],
                   rhat_walkers=split_rhat(x[:, :, i]), mean=v.mean(), sd=v.std(ddof=1), median=med,
                   ci95_low=lo, ci95_high=hi)
        if lower is not None and upper is not None:
            row['mass_near_lower_bound'] = float(np.mean(v < lower[i] + 0.05 * (upper[i] - lower[i])))
        rows.append(row)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table.to_csv(args.out, index=False, float_format='%.5g')
    corr = pd.DataFrame(np.corrcoef(x[:, :, :args.n_model_params].reshape(-1, args.n_model_params).T),
                        index=labels[:args.n_model_params], columns=labels[:args.n_model_params])
    corr.to_csv(args.out.replace('.csv', '_correlation.csv'), float_format='%.3f')
    print(f'{steps} steps x {walkers} walkers after burn-in')
    print(table.to_string(index=False))
    print('\nPosterior correlation of the model parameters:')
    print(corr.round(3).to_string())


if __name__ == '__main__':
    main()
