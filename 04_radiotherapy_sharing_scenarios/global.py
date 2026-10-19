"""Scenarios V1-V8 for sharing the radiotherapy parameters (alpha_rad, beta_rad, delay) between
radiation_sensitive and radiation_resistant, for the selected delayed exponential decay model.

Growth (r, cc) is inherited and fixed; the effect is anchored at the dose days. Walkers start in a
ball around a MAP estimate, stuck walkers are removed, the HDF5 backend is resumable, and seven
selection metrics are computed.

Settings: CALIB_* environment variables (see README.md), e.g. CALIB_SCENARIOS=1,2,...,8,
CALIB_DOSE_DAYS=0,1,2,3,4,5, CALIB_DOSE=2, CALIB_GROWTH_SOURCE=fixed.
"""
import gc
import os
from multiprocessing import cpu_count, get_context

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import numpy as np

import _path_setup  # noqa: F401
from initialization import initial_positions
from walkers import drop_stuck
from postprocess import compress_h5_inplace
from model_selection import normalize_metrics

from data_loader_rad import load_radiation_data, parse_dose_days_env
from bounds_rad import find_max_initial_condition
from growth_params import load_growth_params
from scenarios_rad import SCENARIOS, WINNER_MODEL, scenario_id, describe
from bounds_scen_rad import define_scenario_bounds_labels
from inference_scen_rad import log_probability_shared
from metrics_scen_rad import compare_scenarios, print_comparison_table

os.makedirs('./Output_Calibration', exist_ok=True)
os.makedirs('./reinitiate_files', exist_ok=True)

GROUPS = ['radiation_sensitive', 'radiation_resistant']
# Dose schedule from CALIB_DOSE_DAYS (thesis: 0,1,2,3,4,5).
DOSE_DAYS = parse_dose_days_env(os.environ.get('CALIB_DOSE_DAYS'))
full_data = load_radiation_data(data_dir=_path_setup.DATA_DIR, dose_days=DOSE_DAYS)
max_ic = find_max_initial_condition(full_data)


def _parse_scenarios(env_value, model_extension):
    valid = set(SCENARIOS[model_extension])
    if not env_value:
        return sorted(valid)
    req = [int(x) for x in env_value.split(',') if x.strip()]
    bad = [s for s in req if s not in valid]
    if bad:
        raise ValueError(f'Invalid scenarios: {bad}. Valid: {sorted(valid)}')
    return sorted(set(req))


if __name__ == '__main__':
    for g, mice in full_data.items():
        print(f'{g}: {len(mice)} mice')

    MODEL = os.environ.get('CALIB_MODEL', WINNER_MODEL)
    CHAIN_SIZE = int(os.environ.get('CALIB_CHAIN_SIZE', '1000'))
    USE_POOL = os.environ.get('CALIB_USE_POOL', '1') == '1'
    N_CORES = int(os.environ.get('CALIB_N_CORES', str(cpu_count())))
    BACKEND_ONLY = os.environ.get('CALIB_BACKEND_ONLY', '0') == '1'
    COMPRESS_H5 = os.environ.get('CALIB_COMPRESS_H5', '0') == '1'
    WPD = int(os.environ.get('CALIB_WALKERS_PER_DIM', '4'))
    INIT = os.environ.get('CALIB_INIT', 'ball').lower()
    DROP = os.environ.get('CALIB_DROP_STUCK', '1') == '1'
    METRICS = normalize_metrics(os.environ.get('CALIB_METRICS',
                                               'aic,aicc,bic,dic,waic,loo,wbic'))

    TEST = os.environ.get('CALIB_TEST', '0') == '1'
    if TEST:
        CHAIN_SIZE = 50
        BACKEND_ONLY = True
    PREFIX = 'TEST_' if TEST else ''

    scenarios_to_run = _parse_scenarios(os.environ.get('CALIB_SCENARIOS', ''),
                                        MODEL)
    if TEST:
        scenarios_to_run = [4]

    growth = load_growth_params(GROUPS)
    print(f'[config] model={MODEL} scenarios={scenarios_to_run} '
          f'chain={CHAIN_SIZE} init={INIT} backend_only={BACKEND_ONLY}')

    for snum in scenarios_to_run:
        sdef = SCENARIOS[MODEL][snum]
        stem = f'{MODEL}{scenario_id(snum)}'
        print(f'\n--- {stem}: {describe(MODEL, snum)} ---')

        l_bound, u_bound, labels = define_scenario_bounds_labels(
            MODEL, sdef, GROUPS, full_data, max_ic)
        ndim = len(l_bound)
        nwalkers = WPD * ndim
        args = (l_bound, u_bound, MODEL, full_data, GROUPS, sdef, growth)

        filename = f'./reinitiate_files/reinitiate_{PREFIX}{stem}.h5'
        if not os.path.exists(filename):
            print(f'    New HDF5 backend (ndim={ndim}, nwalkers={nwalkers})')
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)
            pos, info = initial_positions(
                nwalkers, l_bound, u_bound, log_probability_shared, args,
                mode=INIT, n_random=int(os.environ.get('CALIB_INIT_NRANDOM', '80')),
                map_maxiter=int(os.environ.get('CALIB_INIT_MAXITER', '30')),
                map_maxfev=int(os.environ.get('CALIB_INIT_MAXFEV', '400')))
            print(f'    [init] {info["mode"]}')
        else:
            backend = emcee.backends.HDFBackend(filename)
            print(f'    HDF5 backend found: {backend.iteration} steps')
            pos = None

        additional = max(0, CHAIN_SIZE - backend.iteration)
        if additional == 0:
            print(f'    [skip] {stem}: {backend.iteration} (target {CHAIN_SIZE})')
            if COMPRESS_H5:
                compress_h5_inplace(filename)
            if BACKEND_ONLY:
                gc.collect(); continue
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            log_probability_shared, args=args,
                                            backend=backend)
        else:
            print(f'    Calibrating: {additional} steps')

            def _go(pool):
                s = emcee.EnsembleSampler(nwalkers, ndim, log_probability_shared,
                                          args=args, pool=pool, backend=backend)
                s.run_mcmc(None if backend.iteration > 0 else pos, additional,
                           progress=True)
                return s

            if USE_POOL:
                with get_context('spawn').Pool(processes=N_CORES) as pool:
                    sampler = _go(pool)
            else:
                sampler = _go(None)
            if COMPRESS_H5:
                compress_h5_inplace(filename)
            if BACKEND_ONLY:
                print(f'    [backend-only] {stem}')
                del sampler, backend; gc.collect(); continue

        flat_ll = sampler.get_log_prob(flat=True)
        best = sampler.get_chain(flat=True)[flat_ll.argmax(axis=0)]
        discard = int(backend.iteration) // 4
        ch3d = sampler.get_chain(discard=discard)
        lp2d = sampler.get_log_prob(discard=discard)
        if DROP:
            ch3d, lp2d, stuck = drop_stuck(ch3d, lp2d)
            if stuck.any():
                print(f'    [walkers] {stuck.sum()}/{len(stuck)} stuck walkers removed')
        thin = max(1, ch3d.shape[0] * ch3d.shape[1] // 10000)
        save_chain = ch3d[::thin].reshape(-1, ndim)
        np.savetxt(f'./Output_Calibration/multi_chain_{stem}.gz', save_chain)
        np.savez(f'./Output_Calibration/multi_ll_pars_{stem}.npz',
                 max_ll=max(flat_ll), pars=best, labels=np.array(labels),
                 n_stuck=int(nwalkers - ch3d.shape[1]))
        print(f'    [ok] max_ll={max(flat_ll):.2f}, {save_chain.shape[0]} samples')
        del sampler, backend; gc.collect()

    if not BACKEND_ONLY:
        df = compare_scenarios('./Output_Calibration', full_data, GROUPS, MODEL,
                               SCENARIOS[MODEL], growth, metrics=METRICS)
        print_comparison_table(df, MODEL, METRICS)
        if not df.empty:
            df.to_csv(f'./Output_Calibration/scenario_selection_{MODEL}.csv',
                      index=False, float_format='%.4f')
