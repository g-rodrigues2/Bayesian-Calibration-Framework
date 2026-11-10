"""Sensitivity of the independent, homoscedastic Gaussian likelihood (Eqs. 4.3-4.5 of the thesis).

Reviewer 1 observes that longitudinal observations of the same mouse are correlated and that the
residual spread may depend on tumor size, so that treating every observation as independent can
understate uncertainty and can also affect the BIC. This script quantifies both effects on the
fitted models, without re-running the model selection:

    1. residuals of the global combined-treatment fit at the posterior median, mouse by mouse;
    2. lag-1 autocorrelation of those residuals within each mouse, pooled over mice;
    3. dependence of the residual spread on the fitted volume (Spearman correlation);
    4. effective sample size of an AR(1) sequence with the pooled autocorrelation,
       n_eff = n (1 - rho) / (1 + rho), and the BIC of the parameter-sharing scenarios recomputed
       with n_eff in place of n, which shows whether the selection survives the correction.

    python likelihood_sensitivity.py

Writes results/{residual_structure.csv,residual_summary.csv,bic_neff.csv}.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _stage_paths  # noqa: F401
from _stage_paths import RESULTS, stage
from data_loader_ri import load_data
from inference_ri import split_theta
from rad_immuno_models import solve_mouse
from thesis_params import upstream_params

OUT = RESULTS
GLOBAL_FIT = os.path.join(stage('06_combined_treatment'), 'Output_Calibration', 'suv',
                          'chain_rad+io_sensitive_SUVLinear.gz')
BIC_TABLES = os.path.join(stage('07_bic_verification'), 'results')
GROUP = 'rad+io_sensitive'
MODEL = '_SUVLinear'


def residuals_per_mouse(mice, upstream):
    """Residuals of each mouse at the posterior median of the global fit."""
    theta = np.median(np.loadtxt(GLOBAL_FIT, ndmin=2), axis=0)
    params, ics, std = split_theta(theta, MODEL, len(mice))
    rows = []
    for mouse, ic in zip(mice, ics):
        obs = mouse['data'][:, 1]
        pred = solve_mouse(MODEL, params, ic, mouse, upstream)
        for t, y, f in zip(mouse['data'][:, 0], obs, pred):
            rows.append(dict(mouse=mouse['name'], day=t, observed=y, fitted=f, residual=y - f))
    return pd.DataFrame(rows), float(std)


def lag1_autocorrelation(residuals):
    """Lag-1 autocorrelation within each mouse, and the pooled estimate.

    The pooled value uses the sums over mice, so that mice with more observations weigh more; the
    mean over mice is reported alongside it as a check that one animal is not driving the result."""
    per_mouse, num, den = [], 0.0, 0.0
    for name, block in residuals.groupby('mouse', sort=False):
        e = block['residual'].to_numpy()
        e = e - e.mean()
        if len(e) < 3:
            continue
        num += float(np.sum(e[:-1] * e[1:]))
        den += float(np.sum(e ** 2))
        per_mouse.append(dict(mouse=name, n_obs=len(e),
                              rho1=float(np.sum(e[:-1] * e[1:]) / np.sum(e ** 2))))
    per_mouse = pd.DataFrame(per_mouse)
    return per_mouse, num / den


def effective_n(n, rho):
    """Observations of an AR(1) sequence worth one independent observation each."""
    return n * (1.0 - rho) / (1.0 + rho)


def bic_with_neff(cells, table_id, rho):
    """Recomputes a scenario table with n_eff in place of n, keeping k and the log-likelihood."""
    table = cells[cells['table'].astype(str) == table_id].copy()
    table['n_eff'] = effective_n(table['n'], rho)
    table['bic_neff'] = table['k'] * np.log(table['n_eff']) - 2 * table['refined_ll']
    for column, source in (('w_neff', 'bic_neff'), ('w_check', 'bic_refined')):
        delta = table[source] - table[source].min()
        weight = np.exp(-0.5 * delta)
        table[column] = weight / weight.sum()
    return table


def main():
    os.makedirs(OUT, exist_ok=True)
    mice = load_data()[GROUP]
    upstream = upstream_params('sensitive')

    residuals, std = residuals_per_mouse(mice, upstream)
    per_mouse, rho = lag1_autocorrelation(residuals)
    spread = spearmanr(residuals['fitted'], residuals['residual'].abs())

    per_mouse.to_csv(os.path.join(OUT, 'residual_structure.csv'), index=False, float_format='%.4f')
    pd.DataFrame([('n_obs', len(residuals)), ('n_mice', len(per_mouse)), ('residual_sd', std),
                  ('rho1_pooled', rho), ('rho1_mean', per_mouse['rho1'].mean()),
                  ('rho1_min', per_mouse['rho1'].min()), ('rho1_max', per_mouse['rho1'].max()),
                  ('spearman_abs_residual_vs_fitted', spread.statistic),
                  ('spearman_pvalue', spread.pvalue), ('neff_factor', (1 - rho) / (1 + rho))],
                 columns=['quantity', 'value']).to_csv(os.path.join(OUT, 'residual_summary.csv'),
                                                       index=False, float_format='%.6g')
    print(f'Residual standard deviation of the fit: {std:.2f} mm3, {len(residuals)} observations, '
          f'{len(per_mouse)} mice')
    print(f'Lag-1 autocorrelation within mice: pooled {rho:.3f}, '
          f'mean over mice {per_mouse["rho1"].mean():.3f}, '
          f'range [{per_mouse["rho1"].min():.3f}, {per_mouse["rho1"].max():.3f}]')
    print(f'Residual spread against fitted volume: Spearman {spread.statistic:.3f} '
          f'(p = {spread.pvalue:.2g})')
    print(f'Effective sample size factor (1-rho)/(1+rho): {(1 - rho) / (1 + rho):.3f}')

    cells = pd.read_csv(os.path.join(BIC_TABLES, 'cells.csv'))
    frames = []
    for table_id, label in (('4', 'Table 4 (control)'), ('5', 'Table 5 (radiotherapy)')):
        table = bic_with_neff(cells, table_id, rho)
        if table.empty:
            continue
        frames.append(table)
        best_n = table.loc[table['bic_refined'].idxmin(), 'column']
        best_eff = table.loc[table['bic_neff'].idxmin(), 'column']
        flag = 'unchanged' if best_n == best_eff else f'CHANGES to {best_eff}'
        print(f'{label}: n = {table["n"].iloc[0]}, n_eff = {table["n_eff"].iloc[0]:.0f}, '
              f'k from {table["k"].min()} to {table["k"].max()}; lowest BIC {best_n} -> {flag}')
    if frames:
        out = pd.concat(frames)[['table', 'row', 'column', 'n', 'n_eff', 'k', 'refined_ll',
                                 'bic_refined', 'w_check', 'bic_neff', 'w_neff']]
        out.to_csv(os.path.join(OUT, 'bic_neff.csv'), index=False, float_format='%.4f')
        print(f'\nWrote {os.path.join(OUT, "bic_neff.csv")}')


if __name__ == '__main__':
    main()
