"""Checks that this pipeline reproduces the results reported in the thesis without re-running MCMC.

The maximum-likelihood samples saved by the original calibrations (thesis results) are evaluated
with the likelihood implemented here and compared with the log-likelihood stored with them; the
strict leave-one-out held-out log-likelihoods are recomputed from the saved training folds and
compared with the saved per-mouse values and with the totals of Table 4.x.

    python validate_legacy.py --legacy <folder with the original Output_Calibration directories>

Expected layout under --legacy (original calibration folders):
    predictions_using_radpars_iopars/Output_Calibration/multi_ll_pars_rad+io_sensitive_RadImmuno.npz
    global_SUV_io_rad+io_SUVfunction/Output_Calibration/multi_ll_pars_rad+io_sensitive_RadImmuno{,2,3,4}.npz
    Testes_finais/io_rad_SUV_leave_one_out/Output_Calibration/      (SUV folds and strictLOO csv)
    Testes_finais/io_rad_global_leave_one_out/Output_Calibration/   (folds without SUV)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from data_loader_ri import load_data
from inference_ri import gaussian_loglik, log_likelihood
from rad_immuno_models import solve_mouse
from thesis_params import upstream_params

GROUP = 'rad+io_sensitive'
LEGACY_NAMES = {'_SUVLinear': 'RadImmuno', '_SUVMichaelisMenten': 'RadImmuno2', '_SUVSatExp': 'RadImmuno4'}


def check(label, stored, recomputed, tol=1e-6):
    ok = abs(stored - recomputed) <= tol * max(1.0, abs(stored))
    print(f'  [{"ok" if ok else "MISMATCH"}] {label}: stored {stored:.6f}, recomputed {recomputed:.6f}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy', required=True)
    root = ap.parse_args().legacy
    mice = load_data()[GROUP]
    up = upstream_params('sensitive')
    results = []

    print('Prediction with fixed parameters (Figure 4.12):')
    z = np.load(os.path.join(root, 'predictions_using_radpars_iopars', 'Output_Calibration',
                             f'multi_ll_pars_{GROUP}_RadImmuno.npz'))
    results.append(check('_RadImmunoFixed', float(z['max_ll']), log_likelihood(z['pars'], '_RadImmunoFixed', mice, up)))

    print('SUV-dependent lysis functions (Table 4.x):')
    suv_dir = os.path.join(root, 'global_SUV_io_rad+io_SUVfunction', 'Output_Calibration')
    for model, legacy in LEGACY_NAMES.items():
        z = np.load(os.path.join(suv_dir, f'multi_ll_pars_{GROUP}_{legacy}.npz'))
        results.append(check(model, float(z['max_ll']), log_likelihood(z['pars'], model, mice, up)))
    print('  [skip] _SUVLogistic: the original code did not implement the logistic function of the thesis table.')

    print('Leave-one-out training folds and strict held-out log-likelihood (Table 4.x):')
    for model, folder, legacy in [('_SUVLinear', 'io_rad_SUV_leave_one_out', 'RadImmuno'),
                                  ('_RadImmunoFixed', 'io_rad_global_leave_one_out', 'RadImmunoSemSUV')]:
        out = os.path.join(root, 'Testes_finais', folder, 'Output_Calibration')
        folds = sorted(glob.glob(os.path.join(out, f'multi_ll_pars_{GROUP}_EXCL-*_{legacy}.npz')))
        heldout = {}
        for path in folds:
            left_out = os.path.basename(path).split('_EXCL-')[1].split(f'_{legacy}')[0]
            train = [m for m in mice if m['name'] != left_out]
            test = next(m for m in mice if m['name'] == left_out)
            z = np.load(path)
            theta = z['pars']
            results.append(check(f'{model} fold without {left_out}', float(z['max_ll']),
                                 log_likelihood(theta, model, train, up)))
            params = tuple(theta[:2]) if model == '_SUVLinear' else ()
            pred = solve_mouse(model, params, test['data'][0, 1], test, up)
            heldout[left_out] = gaussian_loglik(test['data'][1:, 1], pred[1:], float(theta[-1]))
        csv = os.path.join(out, f'loo_metrics_{GROUP}_{legacy}_strictLOO.csv')
        if os.path.exists(csv):
            saved = pd.read_csv(csv).set_index('mouse')['heldout_ll']
            for mouse, value in heldout.items():
                results.append(check(f'{model} held-out LL {mouse}', float(saved[mouse]), value))
        print(f'  {model}: total held-out LL {sum(heldout.values()):.4f}, mean {np.mean(list(heldout.values())):.4f} '
              f'over {len(heldout)} mice')

    print(f'\n{sum(results)}/{len(results)} checks passed.')


if __name__ == '__main__':
    main()
