"""Extends an emcee calibration until explicit convergence criteria are met.

A stage script (`global.py` by default, `--command`) is run repeatedly in backend-only mode with an increasing
CALIB_CHAIN_SIZE (the HDF5 backend is resumed). After each extension the backend is evaluated with the
same burn-in (first quarter of the steps) and stuck-walker removal used for the saved chain:

  - N_post / tau >= --min-tau-multiple (default 50) for every parameter, where tau is the integrated
    autocorrelation time;
  - split R-hat < --max-rhat (default 1.05) for every parameter, with the walkers as chains (for an
    ensemble sampler walkers are not independent, so R-hat is a necessary rather than sufficient
    check).

When both hold, or --max-steps is reached, the script (or --final-command) is run once more without backend-only mode to
write the chain, the estimate and the figures. The diagnostics of every extension are appended to
<stage>/Output_Calibration/convergence_<backend stem>.csv.

    python run_until_converged.py --stage ../05_immunotherapy \\
        --backend reinitiate_files/reinitiate_io_sensitive_CumulativeDose.h5 \\
        --env CALIB_MODELS=_CumulativeDose CALIB_GROWTH_SOURCE=fixed CALIB_WALKERS_PER_DIM=8 \\
        --start 20000 --step 20000 --max-steps 200000

    python run_until_converged.py --stage ../06_combined_treatment \
        --backend reinitiate_files/rad+io_sensitive_SUVLinear.h5 \
        --command "calibrate.py suv --models _SUVLinear --no-figures" \
        --final-command "calibrate.py suv --models _SUVLinear"
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import numpy as np
import pandas as pd

from walkers import drop_stuck


def split_rhat(x):
    """Split R-hat of x with shape (steps, walkers)."""
    half = x.shape[0] // 2
    chains = np.concatenate([x[:half], x[half:2 * half]], axis=1)
    n = chains.shape[0]
    within = chains.var(axis=0, ddof=1).mean()
    between = n * chains.mean(axis=0).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def diagnose(backend_path):
    """(steps, table) of the post-burn-in chain of a backend after stuck-walker removal."""
    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    total = int(backend.iteration)
    chain = backend.get_chain(discard=total // 4)
    log_prob = backend.get_log_prob(discard=total // 4)
    chain, _lp, stuck = drop_stuck(chain, log_prob)
    tau = emcee.autocorr.integrated_time(chain, quiet=True)
    steps = chain.shape[0]
    table = pd.DataFrame({
        'tau': tau,
        'steps_over_tau': steps / tau,
        'rhat': [split_rhat(chain[:, :, i]) for i in range(chain.shape[2])],
    })
    return total, int(stuck.sum()), table


def run_stage(stage, env, chain_size, backend_only, log_path, command='global.py'):
    full_env = dict(os.environ, **env, CALIB_CHAIN_SIZE=str(chain_size),
                    CALIB_BACKEND_ONLY='1' if backend_only else '0')
    with open(log_path, 'a') as log:
        log.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M:%S")} chain_size={chain_size} '
                  f'backend_only={backend_only} ===\n')
        log.flush()
        proc = subprocess.run([sys.executable, *shlex.split(command)], cwd=stage, env=full_env,
                              stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'{command} failed (exit {proc.returncode}); see {log_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, help='stage folder')
    ap.add_argument('--command', default='global.py', help='script and arguments of the extensions')
    ap.add_argument('--final-command', default=None, help='script and arguments of the final run (default: --command)')
    ap.add_argument('--backend', required=True, help='HDF5 backend, relative to the stage folder')
    ap.add_argument('--env', nargs='*', default=[], help='KEY=VALUE settings passed to the script')
    ap.add_argument('--start', type=int, default=20000)
    ap.add_argument('--step', type=int, default=20000)
    ap.add_argument('--max-steps', type=int, default=200000)
    ap.add_argument('--min-tau-multiple', type=float, default=50.0)
    ap.add_argument('--max-rhat', type=float, default=1.05)
    args = ap.parse_args()

    env = dict(kv.split('=', 1) for kv in args.env)
    stage = os.path.abspath(args.stage)
    backend = os.path.join(stage, args.backend)
    stem = os.path.splitext(os.path.basename(backend))[0].replace('reinitiate_', '')
    os.makedirs(os.path.join(stage, 'Output_Calibration'), exist_ok=True)
    os.makedirs(os.path.join(stage, 'logs'), exist_ok=True)
    report = os.path.join(stage, 'Output_Calibration', f'convergence_{stem}.csv')
    log_path = os.path.join(stage, 'logs', f'convergence_{stem}.log')

    target = args.start
    while True:
        run_stage(stage, env, target, backend_only=True, log_path=log_path, command=args.command)
        total, n_stuck, table = diagnose(backend)
        min_mult, max_rhat = table['steps_over_tau'].min(), table['rhat'].max()
        converged = min_mult >= args.min_tau_multiple and max_rhat < args.max_rhat
        row = pd.DataFrame([dict(steps=total, stuck_walkers=n_stuck, tau_max=table['tau'].max(),
                                 min_steps_over_tau=min_mult, max_rhat=max_rhat, converged=converged)])
        row.to_csv(report, mode='a', header=not os.path.exists(report), index=False, float_format='%.4g')
        print(f'[{stem}] steps={total} stuck={n_stuck} tau_max={table["tau"].max():.0f} '
              f'N/tau_min={min_mult:.1f} Rhat_max={max_rhat:.3f} converged={converged}', flush=True)
        if converged or target >= args.max_steps:
            break
        target = min(target + args.step, args.max_steps)

    table.to_csv(report.replace('.csv', '_parameters.csv'), index_label='parameter_index', float_format='%.4g')
    run_stage(stage, env, target, backend_only=False, log_path=log_path,
              command=args.final_command or args.command)
    print(f'[{stem}] outputs written (converged={converged})', flush=True)


if __name__ == '__main__':
    main()
