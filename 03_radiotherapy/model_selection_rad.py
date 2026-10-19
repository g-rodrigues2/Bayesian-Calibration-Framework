"""Radiotherapy model selection: AIC, AICc, BIC, DIC, WAIC, LOO and WBIC.

Applies the functions of `model_selection` (control stage) to the radiotherapy likelihood
(`inference_rad`, with fixed r/cc and the effect anchored at the dose days). Inputs written by global.py:
  multi_chain_{group}{model}.gz  +  multi_ll_pars_{group}{model}.npz  (global mode)
  chain_{group}_{mouse}{model}.gz + ll_pars_{group}_{mouse}{model}.npz (mouse-specific mode)
"""
import _path_setup  # noqa: F401

import os

import numpy as np
import pandas as pd

from model_selection import (
    _METRIC_COLS, bic_weights, compute_aic, compute_aicc, compute_bic,
    compute_dic, compute_loo, compute_waic, compute_wbic, normalize_metrics,
)
from inference_rad import log_likelihood, log_likelihood_pointwise


def n_observations(full_data, group, mouse_index=None):
    mice = full_data[group] if mouse_index is None else [full_data[group][mouse_index]]
    return sum(m['data'].shape[0] for m in mice)


def _pointwise_matrix(chain, model_extension, full_data, group, growth,
                      mouse_index, n_sub=1500):
    if chain.shape[0] > n_sub:
        chain = chain[np.linspace(0, len(chain) - 1, n_sub).astype(int)]
    ll_rows = [li for li in
              (log_likelihood_pointwise(t, model_extension, full_data, group,
                                        growth, mouse_index) for t in chain)
              if li is not None]
    return np.vstack(ll_rows) if len(ll_rows) >= 10 else None


def _paths(results_dir, mode, group, model, mouse=None):
    if mode == 'global':
        d = os.path.join(results_dir, 'Output_Calibration')
        stem = f'{group}{model}'
        return (os.path.join(d, f'multi_ll_pars_{stem}.npz'),
                os.path.join(d, f'multi_chain_{stem}.gz'))
    d = os.path.join(results_dir, 'Output_Calibration_per_mouse')
    stem = f'{group}_{mouse}{model}'
    return (os.path.join(d, f'll_pars_{stem}.npz'),
            os.path.join(d, f'chain_{stem}.gz'))


def compare_models(results_dir, full_data, group, growth, mode='global',
                   models=None, mouse_index=None, metrics=None, n_sub=1500):
    metrics = normalize_metrics(metrics)
    mouse = (full_data[group][mouse_index]['name']
             if mouse_index is not None else None)
    n = n_observations(full_data, group, mouse_index)
    needs_chain = any(m not in ('aic', 'aicc', 'bic') for m in metrics)

    rows = []
    for model in (models or []):
        npz_path, gz_path = _paths(results_dir, mode, group, model, mouse)
        if not os.path.exists(npz_path):
            continue
        npz = np.load(npz_path, allow_pickle=True)
        max_ll = float(npz['max_ll'])
        k = int(len(npz['pars']))
        row = dict(group=group, mouse=mouse or '-', model=model, k=k, n=n,
                   max_ll=max_ll)
        if 'aic' in metrics:
            row['AIC'] = compute_aic(max_ll, k)
        if 'aicc' in metrics:
            row['AICc'] = compute_aicc(max_ll, k, n)
        if 'bic' in metrics:
            row['BIC'] = compute_bic(max_ll, k, n)
        for m in metrics:
            for aux in _METRIC_COLS[m][1]:
                row.setdefault(aux, float('nan'))
        if 'dic' in metrics:
            row['DIC'] = float('nan')
        if 'waic' in metrics:
            row['WAIC'] = float('nan')
        if 'loo' in metrics:
            row['LOO'] = float('nan')
        if 'wbic' in metrics:
            row['WBIC'] = float('nan')

        if needs_chain and os.path.exists(gz_path):
            chain = np.loadtxt(gz_path)
            if chain.ndim == 1:
                chain = chain.reshape(1, -1)
            sub = chain[np.linspace(0, len(chain) - 1,
                                    min(n_sub, len(chain))).astype(int)]
            ll = np.array([log_likelihood(t, model, full_data, group, growth,
                                          mouse_index) for t in sub])
            if 'dic' in metrics:
                ll_bar = log_likelihood(sub.mean(axis=0), model, full_data,
                                        group, growth, mouse_index)
                row['DIC'], row['p_D'] = compute_dic(ll, ll_bar)
            if 'wbic' in metrics:
                row['WBIC'], row['WBIC_ESS'] = compute_wbic(ll, n)
            if 'waic' in metrics or 'loo' in metrics:
                mat = _pointwise_matrix(chain, model, full_data, group, growth,
                                        mouse_index, n_sub)
                if mat is not None:
                    if 'waic' in metrics:
                        row['WAIC'], row['p_waic'], row['waic_high_var_points'] = compute_waic(mat)
                    if 'loo' in metrics:
                        (row['LOO'], row['p_loo'], row['khat_max'],
                         row['khat_bad_frac']) = compute_loo(mat)
        rows.append(row)
    return rows


def run_model_selection(results_dir, full_data, growth, mode='global',
                        groups=None, models=None, metrics=None,
                        save_csv=True, print_table=True, n_sub=1500):
    metrics = normalize_metrics(metrics)
    all_rows = []
    for group in (groups or []):
        targets = [None] if mode == 'global' else range(len(full_data[group]))
        for mi in targets:
            all_rows.extend(compare_models(results_dir, full_data, group, growth,
                                        mode, models, mi, metrics, n_sub))
    if not all_rows:
        if print_table:
            print('[model_selection_rad] no results.')
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    cols = ['group', 'mouse', 'model', 'k', 'n', 'max_ll']
    for m in metrics:
        score, aux = _METRIC_COLS[m]
        for (_key), sub in df.groupby(['group', 'mouse'], sort=False):
            si = sub[score].to_numpy(float)
            df.loc[sub.index, f'd{score}'] = si - np.nanmin(si)
            df.loc[sub.index, f'w_{score}'] = bic_weights(si)
        cols += [score, f'd{score}', f'w_{score}'] + aux
    df = df[[c for c in cols if c in df.columns]]

    if print_table:
        _print(df, metrics)
    if save_csv:
        out = os.path.join(results_dir, 'Output_Calibration',
                           f'model_selection_rad_{mode}.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False, float_format='%.4f')
        if print_table:
            print(f'\nCSV: {out}')
    return df


def _print(df, metrics):
    scores = [_METRIC_COLS[m][0] for m in metrics]
    for (group, mouse), sub in df.groupby(['group', 'mouse'], sort=False):
        target = group if mouse == '-' else f'{group} [{mouse}]'
        sub = sub.sort_values(scores[0])
        print(f'\n=== Radiotherapy: {target}  (n={int(sub["n"].iloc[0])}) ===')
        for i, (_ix, r) in enumerate(sub.iterrows()):
            mark = '*' if i == 0 else ' '
            fields = ' '.join(
                f'{s}={r[s]:.1f}(w{r[f"w_{s}"]:.2f})'
                for s in scores if np.isfinite(r.get(s, np.nan)))
            flags = ''
            if 'p_D' in r and np.isfinite(r['p_D']) and r['p_D'] < 0:
                flags += ' p_D<0!'
            if 'khat_max' in r and np.isfinite(r['khat_max']) and r['khat_max'] > 0.7:
                flags += ' khat>0.7!'
            print(f'{mark} {r["model"]:<22} k={int(r["k"])} {fields}{flags}')
