"""Parameter sharing between the control groups (TNBC).

Question: can the growth parameters (`r`, `cc`) of the selected growth model be shared between
control_sensitive and control_resistant without a meaningful loss of fit? Each scenario sets each
parameter as 'global' (one value for both groups) or 'group' (one value per group).

For each model in CALIB_MODELS and each scenario in CALIB_SCENARIOS:
  - builds the multi-group theta (scenarios.SCENARIOS)
  - runs emcee with log_probability_shared (resumable HDF5 backend)
  - initializes the walkers in a ball around a MAP estimate and removes stuck walkers
  - writes the chain and the maximum-likelihood sample to Output_Calibration/

Settings (environment variables):
  CALIB_MODELS=_exp,_log         CALIB_SCENARIOS=1,2,3,4
  CALIB_CHAIN_SIZE=30000         CALIB_WALKERS_PER_DIM=4
  CALIB_INIT=ball                CALIB_DROP_STUCK=1
  CALIB_METRICS=aic,aicc,bic,dic,waic,loo,wbic
  CALIB_USE_POOL / CALIB_N_CORES / CALIB_BACKEND_ONLY / CALIB_COMPRESS_H5
  CALIB_TEST=0
"""
import gc
import os
from multiprocessing import cpu_count, get_context

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import numpy as np

# Adds 01_control_growth/ to sys.path (before importing its modules).
import _path_setup
from data_loader import load_control_data
from bounds import find_max_initial_condition
from postprocess import compress_h5_inplace
from buffered_backend import BufferedHDFBackend
from initialization import initial_positions
from walkers import drop_stuck

from bounds_shared import define_shared_bounds_labels
from inference_shared import log_probability_shared
from metrics_shared import compare_scenarios, print_comparison_table
from scenarios import SCENARIOS, scenario_id


os.makedirs('./Output_Calibration', exist_ok=True)
os.makedirs('./reinitiate_files', exist_ok=True)

GROUPS = ['control_sensitive', 'control_resistant']
full_data = load_control_data(data_dir=_path_setup.DATA_DIR)
max_ic = find_max_initial_condition(full_data)[0]


def _parse_scenarios(env_value, model_extension):
    """Comma-separated list -> sorted scenario numbers, validated against SCENARIOS[model]."""
    valid = set(SCENARIOS[model_extension].keys())
    if not env_value:
        return sorted(valid)
    requested = [int(x) for x in env_value.split(',') if x.strip()]
    invalid = [s for s in requested if s not in valid]
    if invalid:
        raise ValueError(f'Invalid scenarios for {model_extension}: {invalid}. '
                         f'Valid: {sorted(valid)}')
    return sorted(set(requested))


if __name__ == '__main__':
    for g, mice in full_data.items():
        print(f'{g}: {len(mice)} mice -> {[m["name"] for m in mice]}')

    MODELS_TO_FIT   = os.environ.get('CALIB_MODELS', '_exp,_log').split(',')
    CHAIN_SIZE      = int(os.environ.get('CALIB_CHAIN_SIZE', '1000'))
    USE_POOL        = os.environ.get('CALIB_USE_POOL', '1') == '1'
    N_CORES         = int(os.environ.get('CALIB_N_CORES', str(cpu_count())))
    BACKEND_ONLY    = os.environ.get('CALIB_BACKEND_ONLY', '0') == '1'
    COMPRESS_H5     = os.environ.get('CALIB_COMPRESS_H5', '0') == '1'
    COMPRESS_LEVEL  = int(os.environ.get('CALIB_COMPRESS_LEVEL', '4'))
    WALKERS_PER_DIM = int(os.environ.get('CALIB_WALKERS_PER_DIM', '4'))
    INIT_MODE       = os.environ.get('CALIB_INIT', 'ball').lower()
    DROP_STUCK      = os.environ.get('CALIB_DROP_STUCK', '1') == '1'
    from metrics_shared import normalize_metrics
    METRICS         = normalize_metrics(os.environ.get(
        'CALIB_METRICS', 'aic,aicc,bic,dic,waic,loo,wbic'))

    TEST_MODE = os.environ.get('CALIB_TEST', '0') == '1'
    if TEST_MODE:
        MODELS_TO_FIT = ['_exp']
        CHAIN_SIZE = 50
        BACKEND_ONLY = True
        print("[TEST MODE] _exp, scenarios 1 and 4, 50 steps, backend only; "
              "files prefixed with TEST_.")
    PREFIX = 'TEST_' if TEST_MODE else ''

    from models import SOLVER
    print(f"[config] models={MODELS_TO_FIT} groups={GROUPS} chain={CHAIN_SIZE} "
          f"pool={USE_POOL} n_cores={N_CORES} walkers_per_dim={WALKERS_PER_DIM} "
          f"init={INIT_MODE} drop_stuck={DROP_STUCK} solver={SOLVER} "
          f"backend_only={BACKEND_ONLY}" + (' [TEST]' if TEST_MODE else ''))

    for model_extension in MODELS_TO_FIT:
        scenarios_env = os.environ.get(
            f'CALIB_SCENARIOS_{model_extension[1:].upper()}',
            os.environ.get('CALIB_SCENARIOS', ''))
        scenarios_to_run = _parse_scenarios(scenarios_env, model_extension)
        print(f'\n>>> Model {model_extension}: scenarios {scenarios_to_run}')

        for snum in scenarios_to_run:
            sdef = SCENARIOS[model_extension][snum]
            stem = f'{model_extension}{scenario_id(snum)}'
            print(f'\n--- {stem}  (def = {sdef}) ---')

            l_bound, u_bound, labels = define_shared_bounds_labels(
                model_extension, sdef, GROUPS, full_data, max_ic)
            ndim = len(l_bound)
            nwalkers = WALKERS_PER_DIM * ndim
            sampler_args = (l_bound, u_bound, model_extension, full_data,
                            GROUPS, sdef)

            filename = f'./reinitiate_files/reinitiate_{PREFIX}{stem}.h5'
            if not os.path.exists(filename):
                print('    HDF5 backend not found; creating a new one.')
                backend = BufferedHDFBackend(filename)
                backend.reset(nwalkers, ndim)
                pos, info = initial_positions(
                    nwalkers, l_bound, u_bound, log_probability_shared,
                    sampler_args, mode=INIT_MODE)
                print(f'    [init] {info["mode"]}')
            else:
                print(f'    HDF5 backend found: {filename}')
                backend = BufferedHDFBackend(filename)
                print(f'    Progress: {backend.iteration} steps.')
                pos = None

            additional = max(0, CHAIN_SIZE - backend.iteration)

            if additional == 0:
                print(f'    [skip] {stem}: backend already at {backend.iteration} '
                      f'(target {CHAIN_SIZE})')
                if COMPRESS_H5:
                    compress_h5_inplace(filename, complevel=COMPRESS_LEVEL)
                if BACKEND_ONLY:
                    gc.collect(); continue
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_shared,
                    args=sampler_args, backend=backend)
            else:
                print(f'    Calibrating {stem}: {additional} new steps '
                      f'({backend.iteration} -> {CHAIN_SIZE})')

                def _run(pool):
                    s = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability_shared,
                        args=sampler_args, pool=pool, backend=backend)
                    s.run_mcmc(None if backend.iteration > 0 else pos,
                               additional, progress=True)
                    return s

                try:
                    if USE_POOL:
                        with get_context('spawn').Pool(processes=N_CORES) as pool:
                            print(f'    spawn pool with {N_CORES} cores')
                            sampler = _run(pool)
                    else:
                        sampler = _run(None)
                finally:
                    backend.flush()

                if COMPRESS_H5:
                    compress_h5_inplace(filename, complevel=COMPRESS_LEVEL)
                if BACKEND_ONLY:
                    print(f'    [backend-only] {stem}: HDF5 at {filename}')
                    del sampler, backend; gc.collect(); continue

            # === Chain without burn-in and stuck walkers ===
            flat_ll = sampler.get_log_prob(flat=True)
            best_pars = sampler.get_chain(flat=True)[flat_ll.argmax(axis=0)]
            discard = int(backend.iteration) // 4
            chain3d = sampler.get_chain(discard=discard)
            lp2d = sampler.get_log_prob(discard=discard)
            if DROP_STUCK:
                chain3d, lp2d, stuck = drop_stuck(chain3d, lp2d)
                if stuck.any():
                    print(f'    [walkers] {stuck.sum()}/{len(stuck)} stuck walkers '
                          f'removed')
            thin = max(1, chain3d.shape[0] * chain3d.shape[1] // 10000)
            save_chain = chain3d[::thin].reshape(-1, ndim)

            np.savetxt(f'./Output_Calibration/multi_chain_{stem}.gz', save_chain)
            np.savez(f'./Output_Calibration/multi_ll_pars_{stem}.npz',
                     max_ll=max(flat_ll), pars=best_pars,
                     labels=np.array(labels),
                     n_stuck=int(nwalkers - chain3d.shape[1]))
            print(f'    [ok] {stem}: max_ll={max(flat_ll):.2f}, '
                  f'{save_chain.shape[0]} samples.')
            del sampler, backend; gc.collect()

    # === Scenario comparison ===
    if not BACKEND_ONLY:
        for model_extension in MODELS_TO_FIT:
            df = compare_scenarios(
                out_dir='./Output_Calibration', full_data=full_data,
                groups=GROUPS, model_extension=model_extension,
                scenarios_to_compare=SCENARIOS[model_extension],
                metrics=METRICS)
            print_comparison_table(df, model_extension, METRICS)
            if not df.empty:
                df.to_csv(f'./Output_Calibration/scenario_selection'
                          f'{model_extension}.csv', index=False,
                          float_format='%.4f')
