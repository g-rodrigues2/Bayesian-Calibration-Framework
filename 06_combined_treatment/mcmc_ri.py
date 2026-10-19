"""MCMC runner shared by the analyses: resumable HDF5 backend, ball initialization around a MAP
estimate, removal of stuck walkers, and the chain/estimate files read by the post-processing."""
import os
from multiprocessing import cpu_count, get_context

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import _path_setup  # noqa: F401
from buffered_backend import BufferedHDFBackend
from initialization import initial_positions
from walkers import drop_stuck


def config_from_env():
    return dict(
        chain_size=int(os.environ.get('CALIB_CHAIN_SIZE', '10000')),
        walkers_per_dim=int(os.environ.get('CALIB_WALKERS_PER_DIM', '2')),
        init=os.environ.get('CALIB_INIT', 'ball'),
        n_cores=int(os.environ.get('CALIB_N_CORES', str(cpu_count()))),
        use_pool=os.environ.get('CALIB_USE_POOL', '1') == '1',
        progress=os.environ.get('CALIB_PROGRESS', '1') == '1',
    )


def run_chain(stem, l_bound, u_bound, labels, log_prob, log_prob_args, cfg, out_dir, reinit_dir):
    """Runs (or resumes) the sampler of log_prob(theta, l_bound, u_bound, *log_prob_args) up to
    cfg['chain_size'] steps and writes
    <out_dir>/chain_<stem>.gz (burn-in of 1/4 removed, thinned to <= 10000 samples),
    <out_dir>/estimate_<stem>.npz (max_ll, pars, labels) and <out_dir>/trace_<stem>.png."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reinit_dir, exist_ok=True)
    ndim = len(l_bound)
    nwalkers = cfg['walkers_per_dim'] * ndim
    args = (l_bound, u_bound, *log_prob_args)
    path = os.path.join(reinit_dir, f'{stem}.h5')
    new = not os.path.exists(path)
    backend = BufferedHDFBackend(path)
    pos = None
    if new:
        backend.reset(nwalkers, ndim)
        pos, _ = initial_positions(nwalkers, l_bound, u_bound, log_prob, args, mode=cfg['init'],
                                   n_random=80, map_maxiter=30, map_maxfev=300)
    else:
        nwalkers = backend.shape[0]
    remaining = max(0, cfg['chain_size'] - backend.iteration)
    if remaining:
        def _run(pool):
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args, pool=pool, backend=backend)
            sampler.run_mcmc(pos, remaining, progress=cfg['progress'])
        try:
            if cfg['use_pool']:
                with get_context('spawn').Pool(cfg['n_cores']) as pool:
                    _run(pool)
            else:
                _run(None)
        finally:
            backend.flush()
    if os.environ.get('CALIB_BACKEND_ONLY', '0') == '1':
        return None, None

    total = backend.iteration
    lp = backend.get_log_prob()
    chain = backend.get_chain()
    best = chain.reshape(-1, ndim)[np.argmax(lp.reshape(-1))]
    kept, _, stuck = drop_stuck(chain[total // 4:], lp[total // 4:])
    thin = max(1, kept.shape[0] * kept.shape[1] // 10000)
    np.savetxt(os.path.join(out_dir, f'chain_{stem}.gz'), kept[::thin].reshape(-1, ndim))
    np.savez(os.path.join(out_dir, f'estimate_{stem}.npz'), max_ll=float(lp.max()), pars=best,
             labels=np.array(labels), n_steps=total, n_stuck_walkers=int(stuck.sum()))

    step = max(1, total // 2000)
    fig, axes = plt.subplots(ndim, 1, figsize=(9, 1.6 * ndim), sharex=True, squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(np.arange(0, total, step), chain[::step, :, i], 'k', alpha=0.25, lw=0.6)
        ax.set_ylabel(labels[i])
    axes[-1, 0].set_xlabel('step')
    fig.savefig(os.path.join(out_dir, f'trace_{stem}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return best, float(lp.max())
