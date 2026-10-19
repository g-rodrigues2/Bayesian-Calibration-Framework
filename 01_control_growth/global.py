"""MCMC calibration of the tumor growth models of the control groups (TNBC).

Loads the data (data_loader), defines the prior bounds (bounds), runs emcee with the log-probability
of `inference`, stores the HDF5 backend (optionally compressed), and writes figures (postprocess),
model selection, group comparison and convergence diagnostics.

Two modes, selected with ``CALIB_MODE``:

  global (default)
      One sampler per (model, group): growth parameters shared by the group, one free initial
      condition per mouse and one std; ndim = n_par + n_mice + 1. The pool parallelizes the walkers.

  mouse_specific
      One sampler per (model, group, mouse); ndim = n_par + 2. The pool parallelizes the mice, with a
      serial sampler in each worker.

All settings are CALIB_* environment variables (see README.md and the .slurm files).
"""
import gc
import os
from multiprocessing import cpu_count, get_context

# Must be set before importing emcee (which loads h5py and initializes HDF5). Otherwise reading a
# backend while a calibration writes to it stops the writer with PermissionError (Windows) or EAGAIN (NFS).
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import matplotlib
matplotlib.use('Agg')  # headless: HPC nodes and spawned workers have no display
import matplotlib.pyplot as plt
import numpy as np

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_control_data
from inference import log_probability
from initialization import initial_positions
from walkers import drop_stuck
from posterior_histogram import plot_posterior_histograms
from postprocess import (
    compress_h5_inplace,
    parameter_summary,
    plot_maxll_solution_global,
    plot_maxll_solution_per_mouse,
    posterior_corr_from_chain,
    scatter_both_per_mouse,
)


OUT_GLOBAL = './Output_Calibration'
OUT_MOUSE  = './Output_Calibration_per_mouse'
REINIT_DIR = './reinitiate_files'

# Output folders
os.makedirs(OUT_GLOBAL, exist_ok=True)
os.makedirs(OUT_MOUSE, exist_ok=True)
os.makedirs(REINIT_DIR, exist_ok=True)

# Control data, loaded at import time (spawned workers re-import this module to rebuild full_data)
full_data = load_control_data()
max_ic = find_max_initial_condition(full_data)[0]


# === Helpers shared by both modes ===

def _open_backend(filename, nwalkers, ndim, l_bound, u_bound, cfg=None,
                  sampler_args=()):
    """Opens (or creates) the HDF5 backend. Returns (backend, initial positions or None, nwalkers).

    For an existing backend the number of walkers is taken from the backend rather than from
    CALIB_WALKERS_PER_DIM, so that chains produced with a different setting can be post-processed.

    For a new backend the initial positions come from `initialization`: by default a small ball around
    an approximate MAP estimate instead of a uniform spread over the prior box, which left about 7% of
    the walkers stuck in a high-`std` mode (see walkers.py).
    """
    cfg = cfg or {}
    if not os.path.exists(filename):
        print(f"HDF5 file not found. Creating a new file: {filename}")
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
    else:
        print(f"HDF5 file found: {filename}")
        backend = emcee.backends.HDFBackend(filename)
        print(f"Current progress on the backend: {backend.iteration} iterations.")
        try:
            nwalkers_backend = int(backend.shape[0])
        except AttributeError:
            backend.reset(nwalkers, ndim)          # created but never reset
        else:
            if nwalkers_backend != nwalkers:
                print(f"[info] backend has {nwalkers_backend} walkers "
                      f"(configuration: {nwalkers}); using the backend value.")
            nwalkers = nwalkers_backend

    if backend.iteration > 0:
        return backend, None, nwalkers

    pos, info = initial_positions(
        nwalkers, l_bound, u_bound, log_probability, sampler_args,
        mode=cfg.get('init', 'ball'), scale=cfg.get('init_scale', 1e-3),
        n_random=cfg.get('init_nrandom', 200),
    )
    center = info['center_log_prob']
    print(f"[init] {info['mode']}"
          + (f" | center log_prob = {center:.2f}" if center is not None
             else ""))
    return backend, pos, nwalkers


def _maybe_compress(filename, cfg):
    if not cfg['compress_h5']:
        return
    sb = os.path.getsize(filename)
    compress_h5_inplace(filename, complevel=cfg['compress_level'])
    sa = os.path.getsize(filename)
    print(f"[compress] {os.path.basename(filename)}: "
          f"{sb/1e6:.1f} MB -> {sa/1e6:.1f} MB ({sb/max(sa,1):.1f}x)")


def _save_chain_products(sampler, backend, labels, nwalkers, cfg,
                         chain_gz, npz_path, corner_png, trace_png):
    """Writes the thinned chain, max_ll/pars, corner plot (optional) and trace plot."""
    flat_ll = sampler.get_log_prob(flat=True)
    flat_chain = sampler.get_chain(flat=True)
    best_pars = flat_chain[flat_ll.argmax(axis=0)]

    total_iters = int(backend.iteration)
    discard_burnin = total_iters // 4

    # Stuck walkers (mean log L far below the median walker) are removed before flattening. They do
    # not change max_ll but distort medians, credible intervals, DIC (through theta_bar) and WBIC.
    chain3d = sampler.get_chain(discard=discard_burnin)        # (steps, walkers, ndim)
    lp2d = sampler.get_log_prob(discard=discard_burnin)        # (steps, walkers)
    if cfg.get('drop_stuck', True):
        chain3d, lp2d, stuck = drop_stuck(chain3d, lp2d)
        if stuck.any():
            print(f'[walkers] {stuck.sum()}/{len(stuck)} stuck walkers '
                  f'removed (indices {list(np.flatnonzero(stuck))})')
    nw_kept = chain3d.shape[1]

    kept_samples = chain3d.shape[0] * nw_kept
    scale_thin = max(1, kept_samples // 10000)

    save_chain = chain3d[::scale_thin].reshape(-1, chain3d.shape[2])
    np.savetxt(chain_gz, save_chain)
    np.savez(npz_path, max_ll=max(flat_ll), pars=best_pars, labels=labels,
             n_stuck=int(nwalkers - nw_kept))

    if cfg['make_corner']:
        corner_discard = total_iters // 2
        corner_kept = (total_iters - corner_discard) * nwalkers
        corner_thin = max(1, corner_kept // 5000)
        corner_chain = sampler.get_chain(discard=corner_discard, flat=True,
                                         thin=corner_thin)
        # truths = maximum-likelihood sample
        try:
            import corner        # pulls arviz, so it is imported only when asked for
            corner.corner(corner_chain, labels=labels, truths=best_pars)
            plt.savefig(corner_png, format='png', dpi=300, bbox_inches='tight',
                        pad_inches=0.02)
            plt.close()
        except Exception as exc:          # a corner plot is a diagnostic, not a result
            print(f'[corner] plot skipped: {type(exc).__name__}: {exc}')

    # Trace plot of a thinned chain (avoids MemoryError for long chains)
    full_samples = sampler.get_chain()
    trace_thin = max(1, full_samples.shape[0] // 5000)
    samples = full_samples[::trace_thin]
    del full_samples
    fig, axes = plt.subplots(nrows=len(labels), ncols=1,
                             figsize=(10, len(labels) * 2), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel(f"step (thinned every {trace_thin})")
    plt.savefig(trace_png, format='png', dpi=300, bbox_inches='tight',
                pad_inches=0.02)
    plt.close('all')
    del samples
    gc.collect()


# === Global mode: one sampler per (model, group), pool over walkers ===

def _calibrate_global(model_extension, group, cfg):
    group_size = len(full_data[group])
    l_bound, u_bound, labels = define_bounds_labels(
        model_extension, group_size=group_size, max_ic=max_ic,
        bool_multiple_mice=True,
    )
    ndim = len(l_bound)
    nwalkers = cfg['walkers_per_dim'] * ndim

    filename = f"{REINIT_DIR}/reinitiate_{cfg['prefix']}{group}{model_extension}.h5"
    sampler_args = (l_bound, u_bound, model_extension, full_data, group)
    backend, pos, nwalkers = _open_backend(filename, nwalkers, ndim,
                                           l_bound, u_bound, cfg, sampler_args)

    additional = max(0, cfg['chain_size'] - backend.iteration)

    if additional == 0:
        print(f"[skip] {group}{model_extension}: backend already at "
              f"{backend.iteration} steps (target {cfg['chain_size']})")
        _maybe_compress(filename, cfg)
        if cfg['backend_only']:
            gc.collect()
            return
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=sampler_args, backend=backend)
    else:
        print(f'Calibrating {group} with {model_extension.split("_")[-1]}: '
              f'{additional} new steps '
              f'(backend={backend.iteration} -> target={cfg["chain_size"]})')

        def _run(pool):
            s = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                      args=sampler_args, pool=pool,
                                      backend=backend)
            start = None if backend.iteration > 0 else pos
            s.run_mcmc(start, additional, progress=True)
            return s

        if cfg['use_pool']:
            # 'spawn' avoids OSError [Errno 12] on Linux with a large parent process
            mp_ctx = get_context('spawn')
            print(f"Calibration using {cfg['n_cores']} cores (spawn, walkers)")
            with mp_ctx.Pool(processes=cfg['n_cores']) as pool:
                sampler = _run(pool)
        else:
            print('Calibration in serial mode')
            sampler = _run(None)

        _maybe_compress(filename, cfg)

        if cfg['backend_only']:
            print(f"[backend-only] {group}{model_extension}: HDF5 at {filename}")
            del sampler, backend
            gc.collect()
            return

    stem = f'{group}{model_extension}'
    chain_gz = f'{OUT_GLOBAL}/multi_chain_{stem}.gz'
    npz_path = f'{OUT_GLOBAL}/multi_ll_pars_{stem}.npz'
    _save_chain_products(
        sampler, backend, labels, nwalkers, cfg,
        chain_gz=chain_gz, npz_path=npz_path,
        corner_png=f'{OUT_GLOBAL}/multi_corner_{stem}.png',
        trace_png=f'{OUT_GLOBAL}/multi_chain_{stem}.png',
    )

    plot_maxll_solution_global(
        npz_path, full_data, show=False, nCols=4,
        figure_name=f'{OUT_GLOBAL}/multi_max_ll_{stem}', out_dir=OUT_GLOBAL,
        band=cfg['band'], central=cfg['central'], n_band=cfg['n_band'],
        band_kind=cfg['band_kind'],
    )

    # Credible intervals of the parameters: table and histograms
    parameter_summary(chain_gz, labels, print_table=True, title=stem)
    plot_posterior_histograms(
        chain_gz, labels=labels, npz_path=npz_path,
        figure_basename=f'{OUT_GLOBAL}/posterior_hist_{stem}')

    plt.close('all')
    gc.collect()


# === Mouse-specific mode: one sampler per mouse, pool over mice ===

def _run_one_mouse(task):
    """Calibrates one mouse (top-level so that it can be pickled under 'spawn').

    Returns (mouse_name, final number of steps).
    """
    (model_extension, group, mouse_index, cfg) = task
    mouse_name = full_data[group][mouse_index]['name']

    l_bound, u_bound, labels = define_bounds_labels(
        model_extension, group_size=1, max_ic=max_ic, bool_multiple_mice=False,
    )
    ndim = len(l_bound)
    nwalkers = cfg['walkers_per_dim'] * ndim

    stem = f'{group}_{mouse_name}{model_extension}'
    filename = f"{REINIT_DIR}/reinitiate_{cfg['prefix']}{stem}.h5"
    sampler_args = (l_bound, u_bound, model_extension, full_data, group,
                    mouse_index)
    backend, pos, nwalkers = _open_backend(filename, nwalkers, ndim,
                                           l_bound, u_bound, cfg, sampler_args)

    additional = max(0, cfg['chain_size'] - backend.iteration)

    if additional == 0:
        print(f"[skip] {stem}: backend already at {backend.iteration} steps "
              f"(target {cfg['chain_size']})")
        _maybe_compress(filename, cfg)
        if cfg['backend_only']:
            return mouse_name, int(backend.iteration)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=sampler_args, backend=backend)
    else:
        print(f'[{mouse_name}] {additional} new steps '
              f'(backend={backend.iteration} -> target={cfg["chain_size"]})')
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=sampler_args, backend=backend)
        start = None if backend.iteration > 0 else pos
        # progress bars of concurrent mice would interleave
        sampler.run_mcmc(start, additional, progress=cfg['progress'])
        _maybe_compress(filename, cfg)
        if cfg['backend_only']:
            return mouse_name, int(backend.iteration)

    _save_chain_products(
        sampler, backend, labels, nwalkers, cfg,
        chain_gz=f'{OUT_MOUSE}/chain_{stem}.gz',
        npz_path=f'{OUT_MOUSE}/ll_pars_{stem}.npz',
        corner_png=f'{OUT_MOUSE}/corner_{stem}.png',
        trace_png=f'{OUT_MOUSE}/chain_{stem}.png',
    )
    n_iter = int(backend.iteration)
    del sampler, backend
    gc.collect()
    return mouse_name, n_iter


def _calibrate_mouse_specific(model_extension, group, cfg):
    mice = full_data[group]
    targets = [i for i, m in enumerate(mice)
               if cfg['mice'] is None or m['name'] in cfg['mice']]
    if not targets:
        print(f"[skip] {group}{model_extension}: no mouse selected")
        return

    tasks = [(model_extension, group, i, cfg) for i in targets]
    print(f'Calibrating {group} with {model_extension.split("_")[-1]}: '
          f'{len(tasks)} mice -> {[mice[i]["name"] for i in targets]}')

    if cfg['use_pool'] and len(tasks) > 1:
        n_proc = min(cfg['n_cores'], len(tasks))
        mp_ctx = get_context('spawn')
        print(f'Calibration using {n_proc} cores (spawn, mice)')
        with mp_ctx.Pool(processes=n_proc) as pool:
            results = pool.map(_run_one_mouse, tasks)
    else:
        print('Calibration in serial mode')
        results = [_run_one_mouse(t) for t in tasks]

    for name, n_iter in results:
        print(f'  [done] {group}_{name}{model_extension}: {n_iter} iters')

    if cfg['backend_only']:
        gc.collect()
        return

    plot_maxll_solution_per_mouse(
        full_data, group=group, model_extension=model_extension,
        out_dir=OUT_MOUSE, nCols=4, show=False,
        figure_name=f'{OUT_MOUSE}/max_ll_{group}{model_extension}',
        band=cfg['band'], central=cfg['central'], n_band=cfg['n_band'],
        band_kind=cfg['band_kind'],
    )

    # Per mouse: CCC/PCC with credible intervals, parameter table and histograms
    for i in targets:
        name = mice[i]['name']
        stem = f'{group}_{name}{model_extension}'
        chain_gz = f'{OUT_MOUSE}/chain_{stem}.gz'
        npz_path = f'{OUT_MOUSE}/ll_pars_{stem}.npz'
        if not os.path.exists(chain_gz):
            continue
        posterior_corr_from_chain(
            chain_path=chain_gz, model_extension=model_extension,
            full_data=full_data, group=group, mouse_index=i,
            n_samp=500, seed=42,
        )
        labels = np.load(npz_path, allow_pickle=True)['labels']
        parameter_summary(chain_gz, labels, print_table=True, title=stem)
        plot_posterior_histograms(
            chain_gz, labels=[str(x) for x in labels], npz_path=npz_path,
            figure_basename=f'{OUT_MOUSE}/posterior_hist_{stem}')

    plt.close('all')
    gc.collect()


if __name__ == "__main__":
    for g, mice in full_data.items():
        print(f"{g}: {len(mice)} mice -> {[m['name'] for m in mice]}")

    # --- Configuration from environment variables ---
    MODE = os.environ.get('CALIB_MODE', 'global').lower()
    if MODE not in ('global', 'mouse_specific'):
        raise ValueError(f"invalid CALIB_MODE: {MODE!r} "
                         "(use 'global' or 'mouse_specific')")

    MODELS_TO_FIT = os.environ.get(
        'CALIB_MODELS', '_exp,_mendel,_bert,_lin,_gomp,_surf,_log').split(',')
    CONTROL_GROUPS = os.environ.get(
        'CALIB_GROUPS', 'control_sensitive,control_resistant').split(',')
    _mice_env     = os.environ.get('CALIB_MICE', '').strip()
    MICE          = set(_mice_env.split(',')) if _mice_env else None
    CHAIN_SIZE    = int(os.environ.get('CALIB_CHAIN_SIZE', '1000'))
    USE_POOL      = os.environ.get('CALIB_USE_POOL', '1') == '1'
    N_CORES       = int(os.environ.get('CALIB_N_CORES', str(cpu_count())))
    BACKEND_ONLY  = os.environ.get('CALIB_BACKEND_ONLY', '0') == '1'
    COMPRESS_H5   = os.environ.get('CALIB_COMPRESS_H5', '0') == '1'
    COMPRESS_LEVEL = int(os.environ.get('CALIB_COMPRESS_LEVEL', '4'))
    MAKE_CORNER   = os.environ.get('CALIB_CORNER', '1') == '1'
    WALKERS_PER_DIM = int(os.environ.get('CALIB_WALKERS_PER_DIM', '2'))
    # Walker initialization: 'ball' (default) or 'uniform'
    INIT          = os.environ.get('CALIB_INIT', 'ball').lower()
    INIT_SCALE    = float(os.environ.get('CALIB_INIT_SCALE', '1e-3'))
    INIT_NRANDOM  = int(os.environ.get('CALIB_INIT_NRANDOM', '200'))
    DROP_STUCK    = os.environ.get('CALIB_DROP_STUCK', '1') == '1'
    if INIT not in ('ball', 'uniform'):
        raise ValueError(f"invalid CALIB_INIT: {INIT!r} (use 'ball' or 'uniform')")
    # Bands of the time courses: 95% credible band and/or 95% predictive interval
    MAKE_BAND     = os.environ.get('CALIB_BAND', '1') == '1'
    N_BAND        = int(os.environ.get('CALIB_N_BAND', '200'))
    CENTRAL       = os.environ.get('CALIB_CENTRAL', 'median').lower()
    BAND_KIND     = os.environ.get('CALIB_BAND_KIND', 'both').lower()
    # Model selection at the end
    MAKE_SELECT   = os.environ.get('CALIB_SELECT', '1') == '1'
    # Criteria: a subset of 'aic,aicc,bic,dic,waic,loo,wbic'; CALIB_DIC=0 keeps only the max_ll metrics.
    METRICS       = os.environ.get('CALIB_METRICS', 'aic,aicc,bic,dic,wbic')
    if os.environ.get('CALIB_DIC', '1') != '1':
        METRICS = 'aic,aicc,bic'
    if MAKE_SELECT:
        from model_selection import normalize_metrics
        METRICS = normalize_metrics(METRICS)   # validated early
    # Box plots and Mann-Whitney U test between the groups
    MAKE_BOXPLOT  = os.environ.get('CALIB_BOXPLOT', '1') == '1'
    if CENTRAL not in ('median', 'maxll'):
        raise ValueError(f"invalid CALIB_CENTRAL: {CENTRAL!r} "
                         "(use 'median' or 'maxll')")
    if BAND_KIND not in ('both', 'credible', 'predictive'):
        raise ValueError(f"invalid CALIB_BAND_KIND: {BAND_KIND!r} "
                         "(use 'both', 'credible' or 'predictive')")

    # Test mode: control_resistant is the smallest group (5 mice)
    TEST_MODE = os.environ.get('CALIB_TEST', '0') == '1'
    if TEST_MODE:
        MODELS_TO_FIT  = ['_log']
        CONTROL_GROUPS = ['control_resistant']
        CHAIN_SIZE     = 50
        BACKEND_ONLY   = True
        print("[TEST MODE] _log, control_resistant, 50 steps, backend only; "
              "files prefixed with 'TEST_'.")

    cfg = {
        'prefix':          "TEST_" if TEST_MODE else "",
        'chain_size':      CHAIN_SIZE,
        'use_pool':        USE_POOL,
        'n_cores':         N_CORES,
        'backend_only':    BACKEND_ONLY,
        'compress_h5':     COMPRESS_H5,
        'compress_level':  COMPRESS_LEVEL,
        'make_corner':     MAKE_CORNER,
        'walkers_per_dim': WALKERS_PER_DIM,
        'init':            INIT,
        'init_scale':      INIT_SCALE,
        'init_nrandom':    INIT_NRANDOM,
        'drop_stuck':      DROP_STUCK,
        'mice':            MICE,
        'band':            MAKE_BAND,
        'n_band':          N_BAND,
        'central':         CENTRAL,
        'band_kind':       BAND_KIND,
        # A single sampler in the global mode shows a progress bar; concurrent bars of mice would
        # interleave in the mouse-specific mode with a pool.
        'progress':        not (MODE == 'mouse_specific' and USE_POOL),
    }

    from models import SOLVER, SOLVER_METHOD
    print(f"[config] mode={MODE} models={MODELS_TO_FIT} groups={CONTROL_GROUPS} "
          f"chain={CHAIN_SIZE} pool={USE_POOL} n_cores={N_CORES} "
          f"walkers_per_dim={WALKERS_PER_DIM} solver={SOLVER}"
          + (f"({SOLVER_METHOD})" if SOLVER == 'solve_ivp' else "")
          + f" backend_only={BACKEND_ONLY} compress_h5={COMPRESS_H5}"
          + (f" (level={COMPRESS_LEVEL})" if COMPRESS_H5 else "")
          + f" init={INIT} drop_stuck={DROP_STUCK}"
          + f" corner={MAKE_CORNER}"
          + (f" metrics={','.join(METRICS)}" if MAKE_SELECT else " select=0")
          + f" band={MAKE_BAND}"
          + (f" (n={N_BAND}, kind={BAND_KIND}, central={CENTRAL})" if MAKE_BAND else "")
          + (f" mice={sorted(MICE)}" if MICE else "")
          + (" [TEST]" if TEST_MODE else ""))

    for model_extension in MODELS_TO_FIT:
        print(f"\nSolving model {model_extension}")
        for group in CONTROL_GROUPS:
            print(f"Solving group {group}")
            if MODE == 'global':
                _calibrate_global(model_extension, group, cfg)
            else:
                _calibrate_mouse_specific(model_extension, group, cfg)

        # === After all groups of a model: combined figures and posterior agreement ===
        if BACKEND_ONLY:
            continue

        if MODE == 'mouse_specific':
            # Side-by-side scatter plots pooling the mice of each group
            if len(CONTROL_GROUPS) == 2:
                scatter_both_per_mouse(
                    full_data, CONTROL_GROUPS, model_extension,
                    out_dir=OUT_MOUSE, show=False)
            continue

        plot_maxll_solution_global(
            f'{OUT_GLOBAL}/multi_ll_pars_*{model_extension}.npz',
            full_data, show=False, nCols=4,
            figure_name=f'{OUT_GLOBAL}/multi_max_ll_both{model_extension}',
            out_dir=OUT_GLOBAL,
            band=cfg['band'], central=cfg['central'], n_band=cfg['n_band'],
            band_kind=cfg['band_kind'],
        )
        for group in CONTROL_GROUPS:
            posterior_corr_from_chain(
                chain_path=f'{OUT_GLOBAL}/multi_chain_{group}{model_extension}.gz',
                model_extension=model_extension, full_data=full_data,
                group=group, n_samp=500, seed=42,
            )

    # === Model selection ===
    if not BACKEND_ONLY and MAKE_SELECT:
        try:
            from model_selection import run_model_selection
            print(f'\n=== MODEL SELECTION ({" / ".join(m.upper() for m in METRICS)}) ===')
            run_model_selection(
                results_dir='.', mode=MODE, groups=CONTROL_GROUPS,
                models=MODELS_TO_FIT, full_data=full_data,
                metrics=METRICS, save_csv=True, print_table=True,
            )
        except Exception as _e:
            print(f'[WARN] model_selection failed: {_e.__class__.__name__}: {_e}')

    # === Comparison between groups: box plots and Mann-Whitney U test ===
    if not BACKEND_ONLY and MAKE_BOXPLOT and len(CONTROL_GROUPS) == 2:
        try:
            from group_comparison import run_group_comparison
            print('\n=== GROUP COMPARISON (Mann-Whitney U) ===')
            run_group_comparison(
                results_dir='.', groups=CONTROL_GROUPS, models=MODELS_TO_FIT,
                mode=MODE, full_data=full_data, save_csv=True, save_fig=True,
                print_table=True,
            )
        except Exception as _e:
            print(f'[WARN] group_comparison failed: {_e.__class__.__name__}: {_e}')

    # === Convergence diagnostics ===
    if not BACKEND_ONLY:
        try:
            from convergence_check import run_convergence_check
            print('\n=== CONVERGENCE DIAGNOSTICS ===')
            run_convergence_check(
                results_dir='.', models=MODELS_TO_FIT,
                groups=CONTROL_GROUPS, save_csv=True, print_table=True,
            )
        except Exception as _e:
            print(f'[WARN] convergence_check failed: {_e.__class__.__name__}: {_e}')
