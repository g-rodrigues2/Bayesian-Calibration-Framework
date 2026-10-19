"""Scenario selection (AIC, AICc, BIC, DIC, WAIC, PSIS-LOO, WBIC) for the parameter-sharing
calibration of the control groups.

The functions of `model_selection` (compute_aic/aicc/bic/dic/waic/loo/wbic, bic_weights) are applied to
the shared likelihood of `inference_shared`, so scenarios are compared with the same metrics as the
growth models, including the k-hat diagnostic of LOO.

Inputs written by `global.py`:
  thinned chain without stuck walkers : multi_chain_{model}_s{N}.gz
  max_ll, best sample and labels      : multi_ll_pars_{model}_s{N}.npz

k = len(theta) (shared or group-specific parameters, initial conditions and std).
n = total number of observations of both groups.
"""
import _path_setup  # adds 01_control_growth/ to sys.path  # noqa: F401

import os

import numpy as np
import pandas as pd

from model_selection import (
    _METRIC_COLS, bic_weights, compute_aic, compute_aicc, compute_bic, compute_dic, compute_loo,
    compute_waic, compute_wbic, normalize_metrics,
)
from inference_shared import (
    log_likelihood_shared, log_likelihood_shared_pointwise,
)


def n_observations(full_data, groups):
    """Total number of (t, V) observations over all mice of the groups."""
    return sum(m['data'].shape[0] for g in groups for m in full_data[g])


def _pointwise_matrix(chain, model_extension, full_data, groups, scenario,
                      n_sub=2000):
    """(S, n_obs) matrix of log p(y_i|theta_s) under the shared likelihood."""
    if chain.shape[0] > n_sub:
        idx = np.linspace(0, chain.shape[0] - 1, n_sub).astype(int)
        chain = chain[idx]
    ll_rows = []
    for theta in chain:
        li = log_likelihood_shared_pointwise(theta, model_extension, full_data,
                                             groups, scenario)
        if li is not None:
            ll_rows.append(li)
    return np.vstack(ll_rows) if len(ll_rows) >= 10 else None


def compare_scenarios(out_dir, full_data, groups, model_extension,
                      scenarios_to_compare, metrics=None, n_sub=2000):
    """One row per scenario with the selected metrics, deltas and weights.

    scenarios_to_compare: dict {snum: scenario_def} (from SCENARIOS[model]).
    Returns a DataFrame sorted by the first metric.
    """
    metrics = normalize_metrics(metrics)
    n = n_observations(full_data, groups)
    needs_chain = any(m not in ('aic', 'aicc', 'bic') for m in metrics)

    rows = []
    for snum, sdef in scenarios_to_compare.items():
        stem = f'{model_extension}_s{snum}'
        npz_path = os.path.join(out_dir, f'multi_ll_pars_{stem}.npz')
        gz_path = os.path.join(out_dir, f'multi_chain_{stem}.gz')
        if not os.path.exists(npz_path):
            continue

        npz = np.load(npz_path, allow_pickle=True)
        max_ll = float(npz['max_ll'])
        k = int(len(npz['pars']))

        row = dict(scenario=f's{snum}', model=model_extension, k=k, n=n,
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
            row['WAIC'] = float('nan'); row['waic_high_var_points'] = 0
        if 'loo' in metrics:
            row['LOO'] = float('nan')
        if 'wbic' in metrics:
            row['WBIC'] = float('nan')

        if needs_chain and os.path.exists(gz_path):
            chain = np.loadtxt(gz_path)
            if chain.ndim == 1:
                chain = chain.reshape(1, -1)
            ll = np.array([
                log_likelihood_shared(t, model_extension, full_data, groups, sdef)
                for t in chain[np.linspace(0, len(chain) - 1,
                                           min(n_sub, len(chain))).astype(int)]
            ])
            if 'dic' in metrics:
                ll_bar = log_likelihood_shared(chain.mean(axis=0),
                                               model_extension, full_data,
                                               groups, sdef)
                row['DIC'], row['p_D'] = compute_dic(ll, ll_bar)
            if 'wbic' in metrics:
                row['WBIC'], row['WBIC_ESS'] = compute_wbic(ll, n)
            if 'waic' in metrics or 'loo' in metrics:
                mat = _pointwise_matrix(chain, model_extension, full_data,
                                        groups, sdef, n_sub)
                if mat is not None:
                    if 'waic' in metrics:
                        row['WAIC'], row['p_waic'], row['waic_high_var_points'] = \
                            compute_waic(mat)
                    if 'loo' in metrics:
                        row['LOO'], row['p_loo'], row['khat_max'], \
                            row['khat_bad_frac'] = compute_loo(mat)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = ['scenario', 'model', 'k', 'n', 'max_ll']
    for m in metrics:
        score, aux = _METRIC_COLS[m]
        s = df[score].to_numpy(float)
        df[f'd{score}'] = s - np.nanmin(s)
        df[f'w_{score}'] = bic_weights(s)
        cols += [score, f'd{score}', f'w_{score}'] + aux
    return df[[c for c in cols if c in df.columns]].sort_values(
        _METRIC_COLS[metrics[0]][0]).reset_index(drop=True)


def print_comparison_table(df, model_extension, metrics=None):
    """Prints the scenario comparison of a model and marks the best scenario."""
    if df.empty:
        print(f'[compare] {model_extension}: no calibrated scenario yet.')
        return
    metrics = normalize_metrics(metrics)
    scores = [_METRIC_COLS[m][0] for m in metrics]
    print(f'\n=== Parameter sharing: {model_extension} '
          f'(n={int(df["n"].iloc[0])}) ===')

    hdr = f'{"":2s}{"scenario":<9s}{"k":>3s}{"max_ll":>11s}'
    for s in scores:
        hdr += f'{s:>10s}{"d"+s:>8s}{"w_"+s:>8s}'
        if s == 'DIC':
            hdr += f'{"p_D":>9s}'
        if s == 'LOO':
            hdr += f'{"khat":>7s}'
    print(hdr)

    for i, r in df.iterrows():
        line = f'{"*" if i == 0 else " "} {r["scenario"]:<8s}{int(r["k"]):>3d}{r["max_ll"]:>11.2f}'
        for s in scores:
            v = lambda c, w, p=2: (f'{r[c]:>{w}.{p}f}'
                                   if c in r and np.isfinite(r[c]) else f'{"-":>{w}s}')
            line += v(s, 10) + v(f'd{s}', 8) + v(f'w_{s}', 8, 3)
            if s == 'DIC':
                flag = '!' if 'p_D' in r and np.isfinite(r['p_D']) and r['p_D'] < 0 else ' '
                line += v('p_D', 8) + flag
            if s == 'LOO':
                kh = r.get('khat_max', float('nan'))
                flag = '!' if np.isfinite(kh) and kh > 0.7 else ' '
                line += (f'{kh:>6.2f}' if np.isfinite(kh) else f'{"-":>6s}') + flag
        print(line)

    best = df.iloc[0]
    print(f'  -> best {scores[0]}: scenario {best["scenario"]} '
          f'({_describe_scenario(model_extension, best["scenario"])})')


def _describe_scenario(model_extension, scenario_tag):
    """Describes a scenario in terms of shared and group-specific parameters."""
    from scenarios import SCENARIOS
    snum = int(scenario_tag.lstrip('s'))
    sdef = SCENARIOS[model_extension][snum]
    parts = [f'{p}={'shared' if k == 'global' else 'group-specific'}'
             for p, k in sdef.items()]
    return ', '.join(parts)
