"""Calibration of the radiotherapy models (TNBC).

Compares radiation-effect models (19 candidates), with growth (r, cc) inherited from the control
groups (logistic model, fixed) and the effect anchored at the dose days of each mouse (2 Gy per
fraction). Two modes (CALIB_MODE):

  global (default)  -- one sampler per (model, group); radiotherapy parameters shared by the group,
                       one initial condition per mouse and one std.
  mouse_specific    -- one sampler per (model, group, mouse).

Walkers are initialized in a ball around a MAP estimate (CALIB_INIT), stuck walkers are removed
(CALIB_DROP_STUCK), the HDF5 backend is resumable, and seven selection metrics are computed.

Settings: CALIB_* environment variables (see README.md). CALIB_MODELS defaults to the 19 models.
"""
import gc
import os
from multiprocessing import cpu_count, get_context

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import matplotlib
matplotlib.use('Agg')
import numpy as np

import matplotlib.pyplot as plt

import _path_setup  # noqa: F401  (adds the control stages to sys.path)
from initialization import initial_positions
from walkers import drop_stuck
from postprocess import compress_h5_inplace, parameter_summary
from posterior_histogram import plot_posterior_histograms
from postprocess_rad import (
    plot_maxll_rad, posterior_corr_rad, scatter_both_rad,
)
from prior_predictive_rad import plot_prior_predictive

from radiation_models import ALL_MODELS, SOLVER
from bounds_rad import define_bounds_labels, find_max_initial_condition
from data_loader_rad import load_radiation_data, parse_dose_days_env
from growth_params import load_growth_params
from inference_rad import log_probability, DOSE

OUT_GLOBAL = './Output_Calibration'
OUT_MOUSE = './Output_Calibration_per_mouse'
REINIT = './reinitiate_files'
for _d in (OUT_GLOBAL, OUT_MOUSE, REINIT):
    os.makedirs(_d, exist_ok=True)

# Dose schedule from CALIB_DOSE_DAYS (e.g. "0,1,2,3,4,5"), applied to all mice.
DOSE_DAYS = parse_dose_days_env(os.environ.get('CALIB_DOSE_DAYS'))
full_data = load_radiation_data(dose_days=DOSE_DAYS)
max_ic = find_max_initial_condition(full_data)


def _open_backend(filename, nwalkers, ndim, l_bound, u_bound, cfg, sampler_args):
    if not os.path.exists(filename):
        print(f'New HDF5 backend: {filename}')
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
    else:
        backend = emcee.backends.HDFBackend(filename)
        print(f'HDF5 backend found: {filename} ({backend.iteration} steps)')
        try:
            nwalkers = int(backend.shape[0])
        except AttributeError:
            backend.reset(nwalkers, ndim)          # created but never reset

    if backend.iteration > 0:
        return backend, None, nwalkers

    pos, info = initial_positions(
        nwalkers, l_bound, u_bound, log_probability, sampler_args,
        mode=cfg['init'], n_random=cfg['init_nrandom'],
        map_maxiter=cfg['init_maxiter'], map_maxfev=cfg['init_maxfev'])
    print(f'    [init] {info["mode"]}')
    return backend, pos, nwalkers


def _save(sampler, backend, labels, nwalkers, cfg, chain_gz, npz_path,
          corner_png=None, trace_png=None):
    flat_ll = sampler.get_log_prob(flat=True)
    best = sampler.get_chain(flat=True)[flat_ll.argmax(axis=0)]
    total = int(backend.iteration)
    discard = total // 4
    chain3d = sampler.get_chain(discard=discard)
    lp2d = sampler.get_log_prob(discard=discard)
    if cfg['drop_stuck']:
        chain3d, lp2d, stuck = drop_stuck(chain3d, lp2d)
        if stuck.any():
            print(f'    [walkers] {stuck.sum()}/{len(stuck)} stuck walkers removed')
    thin = max(1, chain3d.shape[0] * chain3d.shape[1] // 10000)
    save_chain = chain3d[::thin].reshape(-1, chain3d.shape[2])
    np.savetxt(chain_gz, save_chain)
    np.savez(npz_path, max_ll=max(flat_ll), pars=best, labels=np.array(labels),
             n_stuck=int(nwalkers - chain3d.shape[1]))

    if cfg['make_corner'] and corner_png:
        cd = total // 2
        ck = (total - cd) * nwalkers
        cc_chain = sampler.get_chain(discard=cd, flat=True,
                                     thin=max(1, ck // 5000))
        try:
            import corner        # pulls arviz, so it is imported only when asked for
            corner.corner(cc_chain, labels=labels, truths=best)
            plt.savefig(corner_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()
        except Exception as exc:          # a corner plot is a diagnostic, not a result
            print(f'[corner] plot skipped: {type(exc).__name__}: {exc}')

    if trace_png:
        full = sampler.get_chain()
        samples = full[::max(1, full.shape[0] // 5000)]
        fig, axes = plt.subplots(len(labels), 1, figsize=(10, len(labels) * 2),
                                 sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for i in range(len(labels)):
            axes[i].plot(samples[:, :, i], 'k', alpha=0.3)
            axes[i].set_xlim(0, len(samples))
            axes[i].set_ylabel(labels[i])
            axes[i].yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel('step (thinned)')
        plt.savefig(trace_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close('all')

    print(f'    [ok] max_ll={max(flat_ll):.2f}, {save_chain.shape[0]} samples')


def _run_pair(model_extension, group, cfg, growth, mouse_index=None):
    mouse = full_data[group][mouse_index]['name'] if mouse_index is not None else None
    group_size = len(full_data[group])
    l_bound, u_bound, labels = define_bounds_labels(
        model_extension, group_size=group_size, max_ic=max_ic,
        bool_multiple_mice=(mouse_index is None), group=group)
    ndim = len(l_bound)
    nwalkers = cfg['walkers_per_dim'] * ndim
    sampler_args = (l_bound, u_bound, model_extension, full_data, group, growth,
                    mouse_index)

    if mouse_index is None:
        stem = f'{group}{model_extension}'
        chain_gz = f'{OUT_GLOBAL}/multi_chain_{stem}.gz'
        npz_path = f'{OUT_GLOBAL}/multi_ll_pars_{stem}.npz'
        corner_png = f'{OUT_GLOBAL}/multi_corner_{stem}.png'
        trace_png = f'{OUT_GLOBAL}/multi_trace_{stem}.png'
    else:
        stem = f'{group}_{mouse}{model_extension}'
        chain_gz = f'{OUT_MOUSE}/chain_{stem}.gz'
        npz_path = f'{OUT_MOUSE}/ll_pars_{stem}.npz'
        corner_png = f'{OUT_MOUSE}/corner_{stem}.png'
        trace_png = f'{OUT_MOUSE}/trace_{stem}.png'
    filename = f'{REINIT}/reinitiate_{cfg["prefix"]}{stem}.h5'

    backend, pos, nwalkers = _open_backend(filename, nwalkers, ndim, l_bound,
                                           u_bound, cfg, sampler_args)
    additional = max(0, cfg['chain_size'] - backend.iteration)

    if additional == 0:
        print(f'    [skip] {stem}: {backend.iteration} (target {cfg["chain_size"]})')
        if cfg['compress_h5']:
            compress_h5_inplace(filename, complevel=cfg['compress_level'])
        if cfg['backend_only']:
            gc.collect(); return
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=sampler_args, backend=backend)
    else:
        print(f'    Calibrating {stem}: {additional} steps '
              f'({backend.iteration} -> {cfg["chain_size"]})')

        def _go(pool):
            s = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                      args=sampler_args, pool=pool, backend=backend)
            s.run_mcmc(None if backend.iteration > 0 else pos, additional,
                       progress=cfg['progress'])
            return s

        if cfg['use_pool']:
            with get_context('spawn').Pool(processes=cfg['n_cores']) as pool:
                sampler = _go(pool)
        else:
            sampler = _go(None)
        if cfg['compress_h5']:
            compress_h5_inplace(filename, complevel=cfg['compress_level'])
        if cfg['backend_only']:
            del sampler, backend; gc.collect(); return

    _save(sampler, backend, labels, nwalkers, cfg, chain_gz, npz_path,
          corner_png, trace_png)
    del sampler, backend; gc.collect()


def _run_one_mouse(task):
    """Calibrates one mouse inside a worker of the pool over mice, so the sampler is serial (a nested
    pool would fail with 'daemonic processes are not allowed to have children')."""
    model_extension, group, mouse_index, cfg, growth = task
    cfg = dict(cfg, use_pool=False, progress=False)
    _run_pair(model_extension, group, cfg, growth, mouse_index)
    return full_data[group][mouse_index]['name']


if __name__ == '__main__':
    for g, mice in full_data.items():
        print(f'{g}: {len(mice)} mice')

    MODE = os.environ.get('CALIB_MODE', 'global').lower()
    MODELS = os.environ.get('CALIB_MODELS', ','.join(ALL_MODELS)).split(',')
    GROUPS = os.environ.get('CALIB_GROUPS',
                            'radiation_sensitive,radiation_resistant').split(',')
    cfg = dict(
        chain_size=int(os.environ.get('CALIB_CHAIN_SIZE', '1000')),
        use_pool=os.environ.get('CALIB_USE_POOL', '1') == '1',
        n_cores=int(os.environ.get('CALIB_N_CORES', str(cpu_count()))),
        backend_only=os.environ.get('CALIB_BACKEND_ONLY', '0') == '1',
        compress_h5=os.environ.get('CALIB_COMPRESS_H5', '0') == '1',
        compress_level=int(os.environ.get('CALIB_COMPRESS_LEVEL', '4')),
        walkers_per_dim=int(os.environ.get('CALIB_WALKERS_PER_DIM', '2')),
        make_corner=os.environ.get('CALIB_CORNER', '1') == '1',
        band=os.environ.get('CALIB_BAND', '1') == '1',
        n_band=int(os.environ.get('CALIB_N_BAND', '200')),
        prior_pred=os.environ.get('CALIB_PRIOR_PRED', '1') == '1',
        n_prior=int(os.environ.get('CALIB_N_PRIOR', '300')),
        init=os.environ.get('CALIB_INIT', 'ball').lower(),
        drop_stuck=os.environ.get('CALIB_DROP_STUCK', '1') == '1',
        # Budget of the MAP search (the radiotherapy log_prob is expensive).
        init_nrandom=int(os.environ.get('CALIB_INIT_NRANDOM', '80')),
        init_maxiter=int(os.environ.get('CALIB_INIT_MAXITER', '30')),
        init_maxfev=int(os.environ.get('CALIB_INIT_MAXFEV', '300')),
        prefix='',
    )
    cfg['progress'] = not (MODE == 'mouse_specific' and cfg['use_pool'])
    _mice_env = os.environ.get('CALIB_MICE', '').strip()
    MICE = set(_mice_env.split(',')) if _mice_env else None

    if os.environ.get('CALIB_TEST', '0') == '1':
        MODELS = ['_ExponentialDecayDelay']
        GROUPS = ['radiation_resistant']
        cfg['chain_size'] = 50
        cfg['backend_only'] = True
        cfg['prefix'] = 'TEST_'
        print('[TEST MODE] _ExponentialDecayDelay, resistant group, 50 steps')

    METRICS = os.environ.get('CALIB_METRICS', 'aic,aicc,bic,dic,waic,loo,wbic')
    MAKE_SELECT = os.environ.get('CALIB_SELECT', '1') == '1'
    MAKE_FIGS = os.environ.get('CALIB_FIGS', '1') == '1'

    growth = load_growth_params(GROUPS)
    _dd = '[0, 1, 2, 3, 4, 5] (default)' if DOSE_DAYS is None else list(DOSE_DAYS)
    print(f'[config] mode={MODE} models={len(MODELS)} groups={GROUPS} '
          f'chain={cfg["chain_size"]} init={cfg["init"]} solver={SOLVER} '
          f'dose={DOSE}Gy dose_days={_dd}')

    for model_extension in MODELS:
        print(f'\n=== Model {model_extension} ===')
        for group in GROUPS:
            if MODE == 'global':
                _run_pair(model_extension, group, cfg, growth)
            else:
                mice = full_data[group]
                targets = [i for i, m in enumerate(mice)
                           if MICE is None or m['name'] in MICE]
                tasks = [(model_extension, group, i, cfg, growth) for i in targets]
                if cfg['use_pool'] and len(tasks) > 1:
                    with get_context('spawn').Pool(
                            processes=min(cfg['n_cores'], len(tasks))) as pool:
                        for name in pool.map(_run_one_mouse, tasks):
                            print(f'    [done] {group}_{name}{model_extension}')
                else:
                    for t in tasks:
                        _run_one_mouse(t)

        # === Figures of the model ===
        if cfg['backend_only'] or not MAKE_FIGS:
            continue
        out_dir = OUT_GLOBAL if MODE == 'global' else OUT_MOUSE
        for group in GROUPS:
            # Prior predictive check (independent of the chain); dose days in green.
            if cfg['prior_pred']:
                plot_prior_predictive(model_extension, full_data, group, growth,
                                      n_draws=cfg['n_prior'], dose=DOSE,
                                      out_dir=OUT_GLOBAL, max_ic=max_ic)
                plt.close('all')

            # time courses with bands and dose days (green lines)
            plot_maxll_rad(full_data, growth, group, model_extension, out_dir,
                           mode=MODE, nCols=4, band=cfg['band'],
                           n_band=cfg['n_band'], dose=DOSE)
            plt.close('all')

            # CCC/PCC with credible intervals, parameter summary and histograms
            if MODE == 'global':
                stem = f'{group}{model_extension}'
                gz = f'{OUT_GLOBAL}/multi_chain_{stem}.gz'
                npz = f'{OUT_GLOBAL}/multi_ll_pars_{stem}.npz'
            else:
                stem = None
            if MODE == 'global' and os.path.exists(gz):
                posterior_corr_rad(gz, model_extension, growth, full_data, group,
                                   dose=DOSE)
                labels = [str(x) for x in
                          np.load(npz, allow_pickle=True)['labels']]
                parameter_summary(gz, labels, print_table=True, title=stem)
                plot_posterior_histograms(
                    gz, labels=labels, npz_path=npz,
                    figure_basename=f'{OUT_GLOBAL}/posterior_hist_{stem}')
            elif MODE == 'mouse_specific':
                for i, m in enumerate(full_data[group]):
                    st = f"{group}_{m['name']}{model_extension}"
                    gz = f'{OUT_MOUSE}/chain_{st}.gz'
                    npz = f'{OUT_MOUSE}/ll_pars_{st}.npz'
                    if not os.path.exists(gz):
                        continue
                    posterior_corr_rad(gz, model_extension, growth, full_data,
                                       group, mouse_index=i, dose=DOSE)
                    labels = [str(x) for x in
                              np.load(npz, allow_pickle=True)['labels']]
                    parameter_summary(gz, labels, print_table=True, title=st)
                    plot_posterior_histograms(
                        gz, labels=labels, npz_path=npz,
                        figure_basename=f'{OUT_MOUSE}/posterior_hist_{st}')
            plt.close('all')

        # combined scatter plot of the two groups
        if len(GROUPS) == 2:
            scatter_both_rad(full_data, growth, model_extension, out_dir,
                             mode=MODE, groups=tuple(GROUPS), dose=DOSE)
            plt.close('all')
        gc.collect()

    if MAKE_SELECT and not cfg['backend_only']:
        try:
            from model_selection_rad import run_model_selection
            print('\n=== MODEL SELECTION (RADIOTHERAPY) ===')
            run_model_selection('.', full_data, growth, mode=MODE, groups=GROUPS,
                                models=MODELS, metrics=METRICS)
        except Exception as e:
            print(f'[WARN] model selection failed: {e.__class__.__name__}: {e}')
