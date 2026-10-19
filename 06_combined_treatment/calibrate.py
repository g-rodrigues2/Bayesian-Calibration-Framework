"""Calibration of the combined radiotherapy + immunotherapy cohort (Sections 4.5 and 4.6).

    python calibrate.py prediction            # fixed upstream parameters; initial volumes and std only
    python calibrate.py suv                   # the four SUV-dependent lysis functions, BIC comparison
    python calibrate.py suv --models _SUVLinear --no-mcmc   # figures/tables from existing chains

Growth, radiotherapy and de Pillis-Radunskaya parameters are fixed at the values of the previous
stages (thesis_params.py). Outputs are written to Output_Calibration/<analysis>/; resumable HDF5
backends to reinitiate_files/. Environment: CALIB_CHAIN_SIZE (total steps, default 10000),
CALIB_WALKERS_PER_DIM (2), CALIB_INIT (ball | uniform), CALIB_N_CORES.
"""
import argparse
import os

import numpy as np
import pandas as pd

import _path_setup  # noqa: F401
from model_selection import bic_weights, compute_aicc, compute_bic

from bounds_ri import define_bounds_labels
from data_loader_ri import load_data, max_initial_volume
from inference_ri import log_probability
from mcmc_ri import config_from_env, run_chain
from postprocess_ri import figures_from_chain
from rad_immuno_models import SUV_MODELS
from thesis_params import upstream_params

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = 'rad+io_sensitive'
DEFAULT_MODELS = {'prediction': ['_RadImmunoFixed'], 'suv': list(SUV_MODELS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('analysis', choices=['prediction', 'suv'])
    ap.add_argument('--models', default=None, help='comma-separated subset of models')
    ap.add_argument('--no-mcmc', action='store_true', help='only post-process existing chains')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    mice = load_data()[GROUP]
    upstream = upstream_params('sensitive')
    models = args.models.split(',') if args.models else DEFAULT_MODELS[args.analysis]
    out_dir = os.path.join(HERE, 'Output_Calibration', args.analysis)
    reinit_dir = os.path.join(HERE, 'reinitiate_files')
    cfg = config_from_env()
    n_obs = sum(len(m['data']) for m in mice)

    rows = []
    for model in models:
        stem = f'{GROUP}{model}'
        l_bound, u_bound, labels = define_bounds_labels(model, len(mice), max_initial_volume(mice), args.analysis)
        if not args.no_mcmc:
            print(f'Calibrating {stem}: {len(labels)} parameters, {cfg["chain_size"]} steps')
            run_chain(stem, l_bound, u_bound, labels, log_probability, (model, mice, upstream), cfg, out_dir, reinit_dir)
        estimate = np.load(os.path.join(out_dir, f'estimate_{stem}.npz'))
        k, max_ll = len(labels), float(estimate['max_ll'])
        rows.append(dict(model=model, k=k, n=n_obs, max_ll=max_ll, AICc=compute_aicc(max_ll, k, n_obs),
                         BIC=compute_bic(max_ll, k, n_obs)))
        if not args.no_figures:
            figures_from_chain(os.path.join(out_dir, f'chain_{stem}.gz'), model, mice, upstream,
                               os.path.join(out_dir, stem))

    table = pd.DataFrame(rows)
    table['w_BIC'] = bic_weights(table['BIC'])
    table.to_csv(os.path.join(out_dir, 'model_selection.csv'), index=False, float_format='%.4f')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
