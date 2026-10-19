"""Scenario selection (seven metrics) for sharing the radiotherapy parameters, using the functions of
`model_selection` on the multi-group radiotherapy likelihood.
"""
import _path_setup  # noqa: F401

import os

import numpy as np
import pandas as pd

from model_selection import (
    _METRIC_COLS, bic_weights, compute_aic, compute_aicc, compute_bic,
    compute_dic, compute_loo, compute_waic, compute_wbic, normalize_metrics,
)
from inference_scen_rad import (
    log_likelihood_shared, log_likelihood_shared_pointwise,
)
from scenarios_rad import describe


def n_observations(full_data, groups):
    return sum(m['data'].shape[0] for g in groups for m in full_data[g])


def _pointwise(chain, model_extension, full_data, groups, scenario, growth, n_sub):
    if chain.shape[0] > n_sub:
        chain = chain[np.linspace(0, len(chain) - 1, n_sub).astype(int)]
    ll_rows = [li for li in (log_likelihood_shared_pointwise(
        t, model_extension, full_data, groups, scenario, growth) for t in chain)
        if li is not None]
    return np.vstack(ll_rows) if len(ll_rows) >= 10 else None


def compare_scenarios(out_dir, full_data, groups, model_extension,
                      scenarios_to_compare, growth, metrics=None, n_sub=1500):
    metrics = normalize_metrics(metrics)
    n = n_observations(full_data, groups)
    needs_chain = any(m not in ('aic', 'aicc', 'bic') for m in metrics)

    rows = []
    for snum, sdef in scenarios_to_compare.items():
        stem = f'{model_extension}_V{snum}'
        npz_path = os.path.join(out_dir, f'multi_ll_pars_{stem}.npz')
        gz_path = os.path.join(out_dir, f'multi_chain_{stem}.gz')
        if not os.path.exists(npz_path):
            continue
        npz = np.load(npz_path, allow_pickle=True)
        max_ll = float(npz['max_ll'])
        k = int(len(npz['pars']))
        row = dict(scenario=f'V{snum}', sharing=describe(model_extension, snum),
                   k=k, n=n, max_ll=max_ll)
        if 'aic' in metrics:
            row['AIC'] = compute_aic(max_ll, k)
        if 'aicc' in metrics:
            row['AICc'] = compute_aicc(max_ll, k, n)
        if 'bic' in metrics:
            row['BIC'] = compute_bic(max_ll, k, n)
        for m in metrics:
            for aux in _METRIC_COLS[m][1]:
                row.setdefault(aux, float('nan'))
        for score in ('DIC', 'WAIC', 'LOO', 'WBIC'):
            if score.lower() in metrics:
                row[score] = float('nan')

        if needs_chain and os.path.exists(gz_path):
            chain = np.loadtxt(gz_path)
            if chain.ndim == 1:
                chain = chain.reshape(1, -1)
            sub = chain[np.linspace(0, len(chain) - 1,
                                    min(n_sub, len(chain))).astype(int)]
            ll = np.array([log_likelihood_shared(t, model_extension, full_data,
                                                 groups, sdef, growth) for t in sub])
            if 'dic' in metrics:
                ll_bar = log_likelihood_shared(sub.mean(axis=0), model_extension,
                                               full_data, groups, sdef, growth)
                row['DIC'], row['p_D'] = compute_dic(ll, ll_bar)
            if 'wbic' in metrics:
                row['WBIC'], row['WBIC_ESS'] = compute_wbic(ll, n)
            if 'waic' in metrics or 'loo' in metrics:
                mat = _pointwise(chain, model_extension, full_data, groups, sdef,
                                 growth, n_sub)
                if mat is not None:
                    if 'waic' in metrics:
                        row['WAIC'], row['p_waic'], row['waic_high_var_points'] = compute_waic(mat)
                    if 'loo' in metrics:
                        (row['LOO'], row['p_loo'], row['khat_max'],
                         row['khat_bad_frac']) = compute_loo(mat)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ['scenario', 'sharing', 'k', 'n', 'max_ll']
    for m in metrics:
        score, aux = _METRIC_COLS[m]
        s = df[score].to_numpy(float)
        df[f'd{score}'] = s - np.nanmin(s)
        df[f'w_{score}'] = bic_weights(s)
        cols += [score, f'd{score}', f'w_{score}'] + aux
    return df[[c for c in cols if c in df.columns]].sort_values(
        _METRIC_COLS[metrics[0]][0]).reset_index(drop=True)


def print_comparison_table(df, model_extension, metrics=None):
    if df.empty:
        print(f'[compare] {model_extension}: no calibrated scenario.')
        return
    metrics = normalize_metrics(metrics)
    scores = [_METRIC_COLS[m][0] for m in metrics]
    print(f'\n=== Radiotherapy parameter sharing: {model_extension} '
          f'(n={int(df["n"].iloc[0])}) ===')
    for i, r in df.iterrows():
        mark = '*' if i == 0 else ' '
        fields = ' '.join(f'{s}={r[s]:.1f}(w{r[f"w_{s}"]:.2f})'
                          for s in scores if np.isfinite(r.get(s, np.nan)))
        flags = ''
        if 'p_D' in r and np.isfinite(r['p_D']) and r['p_D'] < 0:
            flags += ' p_D<0!'
        if 'khat_max' in r and np.isfinite(r['khat_max']) and r['khat_max'] > 0.7:
            flags += ' khat!'
        print(f'{mark} {r["scenario"]:<4} {fields}{flags}')
        print(f'       [{r["sharing"]}]')
    print(f'  -> best {scores[0]}: {df.iloc[0]["scenario"]} '
          f'({df.iloc[0]["sharing"]})')
