"""Calibration of the immunotherapy models (TNBC).

The 12 candidate formulations of the immunotherapy effect (Table S3.1 of the thesis), with growth
(r, cc) inherited from the control groups and fixed, and the effect anchored at the administration days.

Modes (CALIB_MODE):
  global          -- one sampler per (model, group). Supports a mixed parameterization: CALIB_MS_PARAMS
                     lists the parameters calibrated per mouse (e.g. 'd'); the others are global.
  mouse_specific  -- one sampler per (model, group, mouse).

Walkers start in a ball around a MAP estimate, stuck walkers are removed, the HDF5 backend is resumable,
and seven selection metrics and the figures are produced. Model selection compares every model with a
saved estimate.

CALIB_LOG_PARAMS (e.g. 'd,s') samples the listed parameters on the log scale, with the Jacobian term that
keeps their uniform priors; it improves the mixing of the de Pillis-Radunskaya model, whose effect
d / (1 + s T^lam) defines an approximately linear ridge in (log d, log s, lam). Backends get a '_log-...'
suffix; outputs are written in the original parameters.
"""
import gc
import os
from multiprocessing import cpu_count, get_context

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import _path_setup  # noqa: F401  (adds 01_control_growth/)
from initialization import initial_positions
from walkers import drop_stuck
from postprocess import compress_h5_inplace, parameter_summary
from posterior_histogram import plot_posterior_histograms
from growth_params import load_growth_params

from immuno_models import DOSE_DEFAULT, SOLVER
from bounds_imm import (
    CALIBRATABLE_MODELS, define_bounds_labels,
    find_max_initial_condition,
)
from data_loader_imm import load_immuno_data, parse_dose_days_env
from inference_imm import from_sampling, log_probability, log_probability_logscale, to_sampling
from postprocess_imm import plot_maxll_imm, posterior_corr_imm
from prior_predictive import plot_prior_predictive

OUT_GLOBAL = './Output_Calibration'
OUT_MOUSE = './Output_Calibration_per_mouse'
REINIT = './reinitiate_files'
for _d in (OUT_GLOBAL, OUT_MOUSE, REINIT):
    os.makedirs(_d, exist_ok=True)

DOSE_DAYS = parse_dose_days_env(os.environ.get('CALIB_DOSE_DAYS'))
full_data = load_immuno_data(dose_days=DOSE_DAYS)
max_ic = find_max_initial_condition(full_data)


def _open_backend(filename, nwalkers, ndim, l_bound, u_bound, cfg, args, log_prob=log_probability):
    if not os.path.exists(filename):
        print(f'New HDF5 backend: {os.path.basename(filename)}')
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
    else:
        backend = emcee.backends.HDFBackend(filename)
        print(f'HDF5 backend found ({backend.iteration} steps)')
        try:
            nwalkers = int(backend.shape[0])
        except AttributeError:
            backend.reset(nwalkers, ndim)          # created but never reset

    if backend.iteration > 0:
        return backend, None, nwalkers

    pos, info = initial_positions(
        nwalkers, l_bound, u_bound, log_prob, args,
        mode=cfg['init'], n_random=cfg['init_nrandom'],
        map_maxiter=cfg['init_maxiter'], map_maxfev=cfg['init_maxfev'])
    print(f'    [init] {info["mode"]}')
    return backend, pos, nwalkers


def _save(sampler, backend, labels, nwalkers, cfg, chain_gz, npz_path,
          corner_png, trace_png, log_index=()):
    # chains are stored in sampling coordinates; outputs are written in the original parameters
    back = (lambda x: from_sampling(x, list(log_index))) if log_index else (lambda x: x)
    flat_ll = sampler.get_log_prob(flat=True)
    best = back(sampler.get_chain(flat=True)[flat_ll.argmax(axis=0)])
    if log_index:
        # the stored log-probability includes the Jacobian; max_ll must be the log-likelihood
        flat_ll = flat_ll - sampler.get_chain(flat=True)[:, list(log_index)].sum(axis=1)
        best = back(sampler.get_chain(flat=True)[flat_ll.argmax(axis=0)])
    total = int(backend.iteration)
    discard = total // 4
    raw3d = sampler.get_chain(discard=discard)
    lp2d = sampler.get_log_prob(discard=discard)
    if log_index:
        lp2d = lp2d - raw3d[:, :, list(log_index)].sum(axis=2)
    ch3d = back(raw3d)
    if cfg['drop_stuck']:
        ch3d, lp2d, stuck = drop_stuck(ch3d, lp2d)
        if stuck.any():
            print(f'    [walkers] {stuck.sum()}/{len(stuck)} stuck walkers removed')
    thin = max(1, ch3d.shape[0] * ch3d.shape[1] // 10000)
    save_chain = ch3d[::thin].reshape(-1, ch3d.shape[2])
    np.savetxt(chain_gz, save_chain)
    np.savez(npz_path, max_ll=max(flat_ll), pars=best, labels=np.array(labels),
             n_stuck=int(nwalkers - ch3d.shape[1]))

    if cfg['make_corner']:
        cd = total // 2
        ck = (total - cd) * nwalkers
        try:
            import corner        # pulls arviz, so it is imported only when asked for
            corner.corner(back(sampler.get_chain(discard=cd, flat=True,
                                                 thin=max(1, ck // 5000))),
                          labels=labels, truths=best)
            plt.savefig(corner_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()
        except Exception as exc:          # a corner plot is a diagnostic, not a result
            print(f'[corner] plot skipped: {type(exc).__name__}: {exc}')

    full = back(sampler.get_chain())
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
    mice = full_data[group]
    mouse = mice[mouse_index]['name'] if mouse_index is not None else None
    l_bound, u_bound, labels = define_bounds_labels(
        model_extension, group_size=len(mice), max_ic=max_ic,
        bool_multiple_mice=(mouse_index is None), ms_params=cfg['ms_params'])
    ndim = len(l_bound)
    nwalkers = cfg['walkers_per_dim'] * ndim
    args = (l_bound, u_bound, model_extension, full_data, group, growth,
            mouse_index, cfg['ms_params'], cfg['dose'])
    # optional sampling of selected parameters on the log scale (CALIB_LOG_PARAMS), with the Jacobian
    # term that keeps their uniform priors
    log_index = [labels.index(p) for p in cfg['log_params'] if p in labels]
    log_prob, sample_l, sample_u = log_probability, l_bound, u_bound
    if log_index:
        log_prob = log_probability_logscale
        args = (l_bound, u_bound, log_index) + args[2:]
        sample_l = list(to_sampling(np.asarray(l_bound, float), log_index))
        sample_u = list(to_sampling(np.asarray(u_bound, float), log_index))

    if mouse_index is None:
        stem = f'{group}{model_extension}{cfg["ms_tag"]}'
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
    log_tag = ('_log-' + '-'.join(labels[i] for i in log_index)) if log_index else ''
    filename = f'{REINIT}/reinitiate_{cfg["prefix"]}{stem}{log_tag}.h5'

    backend, pos, nwalkers = _open_backend(filename, nwalkers, ndim, sample_l,
                                           sample_u, cfg, args, log_prob)
    additional = max(0, cfg['chain_size'] - backend.iteration)

    if additional == 0:
        print(f'    [skip] {stem}: {backend.iteration} (target {cfg["chain_size"]})')
        if cfg['compress_h5']:
            compress_h5_inplace(filename)
        if cfg['backend_only']:
            gc.collect(); return
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                        args=args, backend=backend)
    else:
        print(f'    Calibrating {stem}: {additional} steps')

        def _go(pool):
            s = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args,
                                      pool=pool, backend=backend)
            s.run_mcmc(None if backend.iteration > 0 else pos, additional,
                       progress=cfg['progress'])
            return s

        if cfg['use_pool']:
            with get_context('spawn').Pool(processes=cfg['n_cores']) as pool:
                sampler = _go(pool)
        else:
            sampler = _go(None)
        if cfg['compress_h5']:
            compress_h5_inplace(filename)
        if cfg['backend_only']:
            del sampler, backend; gc.collect(); return

    _save(sampler, backend, labels, nwalkers, cfg, chain_gz, npz_path,
          corner_png, trace_png, log_index)
    del sampler, backend; gc.collect()


def _run_one_mouse(task):
    """Runs inside a worker of the pool over mice, so the sampler is serial (a nested pool would fail
    with 'daemonic processes are not allowed to have children')."""
    model_extension, group, mouse_index, cfg, growth = task
    cfg = dict(cfg, use_pool=False, progress=False)
    _run_pair(model_extension, group, cfg, growth, mouse_index)
    return full_data[group][mouse_index]['name']


if __name__ == '__main__':
    for g, mice in full_data.items():
        print(f'{g}: {len(mice)} mice -> {[m["name"] for m in mice]}')

    MODE = os.environ.get('CALIB_MODE', 'global').lower()
    MODELS = os.environ.get('CALIB_MODELS',
                            ','.join(CALIBRATABLE_MODELS)).split(',')
    GROUPS = os.environ.get('CALIB_GROUPS', ','.join(full_data)).split(',')
    # Mixed parameterization: parameters calibrated per mouse (global mode only)
    _ms = os.environ.get('CALIB_MS_PARAMS', '').strip()
    MS_PARAMS = tuple(x.strip() for x in _ms.split(',') if x.strip()) if _ms else ()

    cfg = dict(
        chain_size=int(os.environ.get('CALIB_CHAIN_SIZE', '1000')),
        use_pool=os.environ.get('CALIB_USE_POOL', '1') == '1',
        n_cores=int(os.environ.get('CALIB_N_CORES', str(cpu_count()))),
        backend_only=os.environ.get('CALIB_BACKEND_ONLY', '0') == '1',
        compress_h5=os.environ.get('CALIB_COMPRESS_H5', '0') == '1',
        walkers_per_dim=int(os.environ.get('CALIB_WALKERS_PER_DIM', '2')),
        init=os.environ.get('CALIB_INIT', 'ball').lower(),
        drop_stuck=os.environ.get('CALIB_DROP_STUCK', '1') == '1',
        init_nrandom=int(os.environ.get('CALIB_INIT_NRANDOM', '80')),
        init_maxiter=int(os.environ.get('CALIB_INIT_MAXITER', '30')),
        init_maxfev=int(os.environ.get('CALIB_INIT_MAXFEV', '300')),
        make_corner=os.environ.get('CALIB_CORNER', '1') == '1',
        band=os.environ.get('CALIB_BAND', '1') == '1',
        n_band=int(os.environ.get('CALIB_N_BAND', '200')),
        prior_pred=os.environ.get('CALIB_PRIOR_PRED', '1') == '1',
        n_prior=int(os.environ.get('CALIB_N_PRIOR', '300')),
        dose=float(os.environ.get('CALIB_DOSE', str(DOSE_DEFAULT))),
        ms_params=MS_PARAMS,
        ms_tag=('_ms-' + '-'.join(MS_PARAMS)) if MS_PARAMS else '',
        log_params=tuple(x.strip() for x in os.environ.get('CALIB_LOG_PARAMS', '').split(',') if x.strip()),
        prefix='',
    )
    cfg['progress'] = not (MODE == 'mouse_specific' and cfg['use_pool'])
    _mice = os.environ.get('CALIB_MICE', '').strip()
    MICE = set(_mice.split(',')) if _mice else None

    if os.environ.get('CALIB_TEST', '0') == '1':
        cfg.update(chain_size=50, backend_only=True, prefix='TEST_')
        print('[TEST MODE] 50 steps, backend only')

    METRICS = os.environ.get('CALIB_METRICS', 'aic,aicc,bic,dic,waic,loo,wbic')
    MAKE_SELECT = os.environ.get('CALIB_SELECT', '1') == '1'
    MAKE_FIGS = os.environ.get('CALIB_FIGS', '1') == '1'

    growth = load_growth_params(GROUPS)
    _dd = '[0] (single administration at day 0)' if DOSE_DAYS is None else list(DOSE_DAYS)
    print(f'[config] mode={MODE} models={MODELS} groups={GROUPS} '
          f'chain={cfg["chain_size"]} init={cfg["init"]} solver={SOLVER} '
          f'dose={cfg["dose"]} days={_dd} ms_params={MS_PARAMS or "none"}')

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

        # === Figures ===
        if cfg['backend_only'] or not MAKE_FIGS:
            continue
        out_dir = OUT_GLOBAL if MODE == 'global' else OUT_MOUSE
        for group in GROUPS:
            # Prior predictive check (independent of the chain)
            if cfg['prior_pred']:
                plot_prior_predictive(model_extension, full_data, group, growth,
                                      n_draws=cfg['n_prior'], ms_params=cfg['ms_params'],
                                      dose=cfg['dose'], out_dir=OUT_GLOBAL,
                                      max_ic=max_ic)
                plt.close('all')

            plot_maxll_imm(full_data, growth, group, model_extension, out_dir,
                           mode=MODE, nCols=3, band=cfg['band'],
                           n_band=cfg['n_band'], dose=cfg['dose'],
                           ms_params=cfg['ms_params'],
                           stem_suffix=cfg['ms_tag'] if MODE == 'global' else '')
            plt.close('all')

            if MODE == 'global':
                stem = f'{group}{model_extension}{cfg["ms_tag"]}'
                gz, npz = (f'{OUT_GLOBAL}/multi_chain_{stem}.gz',
                           f'{OUT_GLOBAL}/multi_ll_pars_{stem}.npz')
                if os.path.exists(gz):
                    posterior_corr_imm(gz, model_extension, growth, full_data,
                                       group, dose=cfg['dose'],
                                       ms_params=cfg['ms_params'])
                    labels = [str(x) for x in
                              np.load(npz, allow_pickle=True)['labels']]
                    parameter_summary(gz, labels, print_table=True, title=stem)
                    plot_posterior_histograms(
                        gz, labels=labels, npz_path=npz,
                        figure_basename=f'{OUT_GLOBAL}/posterior_hist_{stem}')
            else:
                for i, m in enumerate(full_data[group]):
                    st = f"{group}_{m['name']}{model_extension}"
                    gz, npz = (f'{OUT_MOUSE}/chain_{st}.gz',
                               f'{OUT_MOUSE}/ll_pars_{st}.npz')
                    if not os.path.exists(gz):
                        continue
                    posterior_corr_imm(gz, model_extension, growth, full_data,
                                       group, mouse_index=i, dose=cfg['dose'])
                    labels = [str(x) for x in
                              np.load(npz, allow_pickle=True)['labels']]
                    parameter_summary(gz, labels, print_table=True, title=st)
                    plot_posterior_histograms(
                        gz, labels=labels, npz_path=npz,
                        figure_basename=f'{OUT_MOUSE}/posterior_hist_{st}')
            plt.close('all')
        gc.collect()

    if MAKE_SELECT and not cfg['backend_only']:
        try:
            from model_selection_imm import run_model_selection
            print('\n=== MODEL SELECTION (IMMUNOTHERAPY) ===')
            run_model_selection('.', full_data, growth, mode=MODE, groups=GROUPS,
                                models=CALIBRATABLE_MODELS, metrics=METRICS,
                                ms_params=cfg['ms_params'])
        except Exception as e:
            print(f'[WARN] model selection failed: {e.__class__.__name__}: {e}')
