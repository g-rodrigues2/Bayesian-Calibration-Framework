"""Exports the MCMC chains reported in Chapter 4 for the data deposit (Zenodo).

Exported chains: the selected models (Appendix table of selected chains) and the leave-one-out analysis
(training folds and held-out chains of both formulations). Each HDF5 backend becomes one compressed file

    <out>/<stage folder>/<backend name>.npz
        chain       (steps/thin, walkers, parameters) float32, every `thin`-th step of the full chain
        log_prob    (steps/thin, walkers) float32
        thin        thinning interval
        n_steps     number of steps of the original chain
        burn_in     burn-in used in the thesis (first quarter of the steps), in original steps
        stuck       (walkers,) walkers removed as stuck in the thesis
        labels      parameter names (in sampling coordinates; d and s of the immunotherapy chain are logs)

Thinning by 10 keeps at least 30 samples per integrated autocorrelation time for every exported chain, so
the convergence diagnostics (steps over tau, split R-hat) computed from the exported files agree with the
thesis tables. Read a file with `read_exported_chain`.

    python export_chains.py --out <folder> [--thin 10]
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '01_control_growth'))

from walkers import find_stuck_walkers  # noqa: E402

SELECTED = [
    ('01_control_growth', 'reinitiate_control_sensitive_log.h5',
     '01_control_growth/Output_Calibration/multi_ll_pars_control_sensitive_log.npz'),
    ('01_control_growth', 'reinitiate_control_resistant_log.h5',
     '01_control_growth/Output_Calibration/multi_ll_pars_control_resistant_log.npz'),
    ('02_control_sharing_scenarios', 'reinitiate__log_s2.h5',
     '02_control_sharing_scenarios/Output_Calibration/multi_ll_pars__log_s2.npz'),
    ('04_radiotherapy_sharing_scenarios', 'reinitiate__ExponentialDecayDelay_V2.h5',
     '04_radiotherapy_sharing_scenarios/Output_Calibration/multi_ll_pars__ExponentialDecayDelay_V2.npz'),
    ('05_immunotherapy', 'reinitiate_io_sensitive_dePillisRadunskayaLaw_log-d-s.h5',
     '05_immunotherapy/Output_Calibration/multi_ll_pars_io_sensitive_dePillisRadunskayaLaw.npz'),
    ('06_combined_treatment', 'rad+io_sensitive_RadImmunoFixed.h5', None),
    ('06_combined_treatment', 'rad+io_sensitive_SUVLinear.h5',
     '06_combined_treatment/Output_Calibration/suv/estimate_rad+io_sensitive_SUVLinear.npz'),
]
LOO_PATTERNS = ('rad+io_sensitive_EXCL-*.h5', 'rad+io_sensitive_PRED-*.h5')


def chain_list():
    items = [(stage, os.path.join(HERE, stage, 'reinitiate_files', name), npz) for stage, name, npz in SELECTED]
    reinit = os.path.join(HERE, '06_combined_treatment', 'reinitiate_files')
    for pattern in LOO_PATTERNS:
        for path in sorted(glob.glob(os.path.join(reinit, pattern))):
            items.append(('06_combined_treatment', path, None))
    return items


def _labels(npz, ndim, name):
    if npz is not None:
        return [str(x) for x in np.load(os.path.join(HERE, npz), allow_pickle=True)['labels']]
    if '_PRED-' in name:
        return ['ic', 'std']
    if name.endswith('_SUVLinear.h5'):
        return ['a', 'b'] + [f'ic{i + 1}' for i in range(ndim - 3)] + ['std']
    if name.endswith('_RadImmunoFitD.h5'):
        return ['d'] + [f'ic{i + 1}' for i in range(ndim - 2)] + ['std']
    return [f'ic{i + 1}' for i in range(ndim - 1)] + ['std']


def export(path, stage, npz, out, thin):
    backend = emcee.backends.HDFBackend(path, read_only=True)
    chain, log_prob = backend.get_chain(), backend.get_log_prob()
    steps = chain.shape[0]
    burn = steps // 4
    stuck = find_stuck_walkers(log_prob[burn:])
    name = os.path.basename(path)
    target = os.path.join(out, stage, name.replace('.h5', '.npz'))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    np.savez_compressed(target, chain=chain[::thin].astype(np.float32), log_prob=log_prob[::thin].astype(np.float32),
                        thin=thin, n_steps=steps, burn_in=burn, stuck=stuck,
                        labels=np.array(_labels(npz, chain.shape[2], name)))
    return dict(file=os.path.relpath(target, out).replace('\\', '/'), steps=int(steps), walkers=int(chain.shape[1]),
                parameters=int(chain.shape[2]), stuck=int(stuck.sum()), megabytes=round(os.path.getsize(target) / 1e6, 1))


def read_exported_chain(path):
    """(chain, log_prob, info) of an exported file; chain and log_prob hold every `thin`-th step."""
    z = np.load(path, allow_pickle=False)
    info = dict(thin=int(z['thin']), n_steps=int(z['n_steps']), burn_in=int(z['burn_in']), stuck=z['stuck'],
                labels=[str(x) for x in z['labels']])
    return z['chain'], z['log_prob'], info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thin', type=int, default=10)
    args = ap.parse_args()
    manifest = []
    for stage, path, npz in chain_list():
        row = export(path, stage, npz, args.out, args.thin)
        manifest.append(row)
        print(row, flush=True)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(f'{len(manifest)} chains, {sum(r["megabytes"] for r in manifest):.1f} MB')


if __name__ == '__main__':
    main()
