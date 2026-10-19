"""Model selection: AIC, AICc, BIC, DIC, WAIC, PSIS-LOO and WBIC, with deltas and weights.

Metrics are selected with `metrics=` (module), `--metrics` (CLI) or `CALIB_METRICS` (environment).
Requesting only the max_ll metrics (aic/aicc/bic) avoids reading the chains.

Metrics (lower is better for all):
  AIC  = 2k - 2 max_ll                              (Akaike)
  AICc = AIC + 2k(k+1)/(n-k-1)                      (small-sample correction; n/k ~ 8-15 here)
  BIC  = k ln(n) - 2 max_ll                         (Schwarz)
  DIC  = D(theta_bar) + 2 p_D,  p_D = D_bar - D(theta_bar),  D = -2 log L
  WAIC = -2 (lppd - p_waic)
  LOO  = -2 elpd_loo (Pareto-smoothed importance sampling)
  WBIC = -2 E_{beta*}[log L],  beta* = 1/ln(n)      (Watanabe, by importance sampling of the chain)

For each metric the difference to the best model and the weight
  w_i = exp(-d_i/2) / sum_j exp(-d_j/2)
are reported (Akaike weights for AIC/AICc, Schwarz weights for BIC). `w_BIC` are BIC weights, not WBIC.

Inputs (written by global.py):
  max_ll, theta   -> .npz  (fallback: maximum of the .h5 backend)
  log_lik samples -> .h5   (log_prob equals log_lik for accepted samples under a uniform prior);
                            fallback: .gz chain with the likelihood recomputed
  n               -> observations of the group (global) or of the mouse (mouse-specific)
  k               -> len(theta), including initial conditions and std

    python model_selection.py                        # global mode
    python model_selection.py --mode mouse_specific
    python model_selection.py --groups control_resistant --no-dic
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable

# Must be set before importing emcee (which loads h5py and initializes HDF5). Otherwise reading a
# backend while a calibration writes to it stops the writer with PermissionError (Windows) or EAGAIN (NFS).
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import numpy as np
import pandas as pd

from scipy.special import logsumexp

from inference import log_likelihood, log_likelihood_pointwise
from walkers import drop_stuck

DEFAULT_MODELS = ['_exp', '_mendel', '_bert', '_lin', '_gomp', '_surf', '_log']
DEFAULT_GROUPS = ['control_sensitive', 'control_resistant']

# Post-processing of the chains
DISCARD_FRAC = 0.25    # first 25% of the steps discarded as burn-in
THIN_TARGET = 5000     # target number of samples per chain after thinning


def n_observations(full_data, group, mouse_index=None):
    """Number of (t, V) observations; mouse_index=None -> whole group."""
    if mouse_index is None:
        return sum(m['data'].shape[0] for m in full_data[group])
    return int(full_data[group][mouse_index]['data'].shape[0])


# === Information criteria ===

def compute_aic(max_ll, k):
    """AIC = 2k - 2 ln L_max (Akaike)."""
    return 2.0 * k - 2.0 * max_ll


def compute_aicc(max_ll, k, n):
    """AICc = AIC + 2k(k+1)/(n-k-1), the small-sample correction of AIC.

    Recommended when n/k < 40 (Burnham & Anderson); here n/k is about 8-15. It converges to AIC for
    n >> k. Returns NaN if n - k - 1 <= 0.
    """
    if n - k - 1 <= 0:
        return float('nan')
    return compute_aic(max_ll, k) + 2.0 * k * (k + 1) / (n - k - 1)


def compute_bic(max_ll, k, n):
    """BIC = k * ln(n) - 2 * ln L_max."""
    if n <= 0:
        return float('nan')
    return k * np.log(n) - 2.0 * max_ll


def compute_dic(log_lik_samples, log_lik_at_mean):
    """DIC = D(theta_bar) + 2 p_D, with p_D = D_bar - D(theta_bar) and D = -2 log L.

    A negative p_D indicates a non-Gaussian (e.g. multimodal) posterior, for which DIC is unreliable.
    Returns (DIC, p_D).
    """
    ll = np.asarray(log_lik_samples, dtype=float)
    ll = ll[np.isfinite(ll)]
    if ll.size < 10 or not np.isfinite(log_lik_at_mean):
        return float('nan'), float('nan')
    D_bar = float((-2.0 * ll).mean())
    D_at_mean = -2.0 * float(log_lik_at_mean)
    p_D = D_bar - D_at_mean
    return D_at_mean + 2.0 * p_D, p_D


def pointwise_loglik(samples, model_extension, full_data, group,
                     mouse_index=None, n_sub=2000):
    """(S, n_obs) matrix of log p(y_i | theta_s), skipping invalid samples.

    Costs one ODE solution per sample; it is the input of WAIC and PSIS-LOO. Returns None if fewer than
    10 samples remain.
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.shape[0] > n_sub:
        idx = np.linspace(0, samples.shape[0] - 1, n_sub).astype(int)
        samples = samples[idx]

    rows = []
    for theta in samples:
        li = log_likelihood_pointwise(theta, model_extension, full_data, group,
                                      mouse_index)
        if li is not None:
            rows.append(li)
    if len(rows) < 10:
        return None
    return np.vstack(rows)


def compute_waic(ll_matrix):
    """WAIC = -2 * (lppd - p_waic), with

        lppd   = sum_i log( (1/S) sum_s p(y_i|theta_s) )
        p_waic = sum_i Var_s[ log p(y_i|theta_s) ]

    It uses the whole posterior, observation by observation. `p_waic_i > 0.4` for some observation
    indicates that WAIC may be biased (Vehtari et al.).

    Var_s is the sample variance (ddof=1), as in eq. 11 of Vehtari, Gelman & Gabry (2017); arviz uses
    ddof=0, and the ratio of the two p_waic values is S/(S-1).

    Returns (WAIC, p_waic, number of observations with p_waic_i > 0.4).
    """
    ll = np.asarray(ll_matrix, float)
    if ll.ndim != 2 or ll.shape[0] < 10:
        return float('nan'), float('nan'), 0
    S = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(S)
    var_i = np.var(ll, axis=0, ddof=1)
    n_high_var = int((var_i > 0.4).sum())
    elpd = float((lppd_i - var_i).sum())
    return -2.0 * elpd, float(var_i.sum()), n_high_var


def compute_loo(ll_matrix, reff=1.0):
    """PSIS-LOO: leave-one-out cross-validation by Pareto-smoothed importance sampling.

        r_is   = 1 / p(y_i | theta_s)       (importance ratios)
        PSIS   = smoothing of the tail of r_is with a generalized Pareto fit
        elpd_i = log( sum_s w_is p(y_i|theta_s) / sum_s w_is )

    The fitted shape `k_hat` of each observation is the diagnostic: with `k_hat > 0.7` the estimate for
    that observation is unreliable, so LOO flags its own failures (unlike DIC).

    Returns (LOO, p_loo, khat_max, fraction of observations with k_hat > 0.7).
    """
    ll = np.asarray(ll_matrix, float)
    if ll.ndim != 2 or ll.shape[0] < 10:
        return float('nan'), float('nan'), float('nan'), float('nan')
    S = ll.shape[0]

    try:
        from arviz.stats import psislw
    except Exception:
        return float('nan'), float('nan'), float('nan'), float('nan')

    # psislw expects (n_obs, n_samples); log ratios = -log p(y_i|theta_s)
    lw, khat = psislw(-ll.T.copy(), reff)
    khat = np.asarray(khat, float).ravel()

    elpd_i = logsumexp(lw + ll.T, axis=-1)      # lw is normalized
    elpd_loo = float(elpd_i.sum())
    lppd = float((logsumexp(ll, axis=0) - np.log(S)).sum())
    p_loo = lppd - elpd_loo
    return -2.0 * elpd_loo, p_loo, float(np.nanmax(khat)), float((khat > 0.7).mean())


def compute_wbic(log_lik_samples, n):
    """WBIC (Watanabe) by importance sampling from the standard posterior chain.

        WBIC = -2 E_{q(theta|beta*)}[log L],  beta* = 1/ln(n),
        q(theta|beta*) ~ p(theta) L(theta)^beta*

    Weights: log w_i = (beta* - 1) log_lik_i, shifted by the maximum. When the spread of log_lik is large
    the weights concentrate on few samples and the estimate is noisy, so the ESS of the weights is also
    returned.

    Returns (WBIC, WBIC_ESS).
    """
    ll = np.asarray(log_lik_samples, dtype=float)
    ll = ll[np.isfinite(ll)]
    if ll.size < 10 or n <= 1:
        return float('nan'), float('nan')
    beta_star = 1.0 / np.log(n)
    log_w = (beta_star - 1.0) * ll
    log_w -= log_w.max()          # numerical stability
    w = np.exp(log_w)
    w_sum = w.sum()
    wbic = -2.0 * float((w * ll).sum() / w_sum)
    ess = float(w_sum ** 2 / (w ** 2).sum())
    return wbic, ess


def bic_weights(scores):
    """Weights from a vector of scores (AIC, AICc, BIC, DIC or WBIC).

        d_i = score_i - min(score);  w_i = exp(-d_i/2) / sum_j exp(-d_j/2)

    Schwarz weights for BIC and Akaike weights for AIC/AICc. `w_BIC` (BIC weights) is not `WBIC`.
    NaN scores get weight 0. The array follows the input order.
    """
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    w = np.zeros_like(s)
    if not finite.any():
        return np.full_like(s, np.nan)
    d = s[finite] - s[finite].min()
    e = np.exp(-0.5 * d)
    w[finite] = e / e.sum()
    return w


def wbic_is_not_bic_weights():  # pragma: no cover - executable documentation
    raise NotImplementedError(
        'Use compute_wbic() for the WBIC of Watanabe and bic_weights() for the '
        'Schwarz weights; they are different quantities.'
    )


# === Loading of calibration outputs ===

def _stem(group, model, mouse=None):
    return f'{group}{model}' if mouse is None else f'{group}_{mouse}{model}'


def _paths(results_dir, mode, group, model, mouse=None):
    """(npz, chain_gz, backend_h5)."""
    if mode == 'global':
        d = os.path.join(results_dir, 'Output_Calibration')
        s = _stem(group, model)
        npz = os.path.join(d, f'multi_ll_pars_{s}.npz')
        gz = os.path.join(d, f'multi_chain_{s}.gz')
    else:
        d = os.path.join(results_dir, 'Output_Calibration_per_mouse')
        s = _stem(group, model, mouse)
        npz = os.path.join(d, f'll_pars_{s}.npz')
        gz = os.path.join(d, f'chain_{s}.gz')
    h5 = os.path.join(results_dir, 'reinitiate_files',
                      f'reinitiate_{_stem(group, model, mouse)}.h5')
    return npz, gz, h5


def load_max_ll(npz_path, h5_path):
    """(max_ll, theta_best) from the .npz file, or from the maximum of the .h5 backend."""
    if os.path.exists(npz_path):
        npz = np.load(npz_path, allow_pickle=True)
        return float(npz['max_ll']), np.asarray(npz['pars'])
    if not os.path.exists(h5_path):
        return None, None
    r = emcee.backends.HDFBackend(h5_path, read_only=True)
    if int(r.iteration) == 0:
        return None, None
    lp = r.get_log_prob(flat=True)
    chain = r.get_chain(flat=True)
    i = int(np.argmax(lp))
    return float(lp[i]), chain[i].copy()


def load_chain_and_loglik(h5_path, gz_path, model_extension, full_data, group,
                          mouse_index=None, drop_stuck_walkers=True):
    """(samples, log_lik) after burn-in and thinning.

    The .h5 backend is preferred: emcee stores log_prob, which equals log_lik under a uniform prior, so no
    ODE has to be solved again. With only the .gz chain (thinned, without log_prob) log_lik is recomputed
    sample by sample (much slower).

    With `drop_stuck_walkers` (default) stuck walkers are removed before flattening, so that
    DIC/WAIC/LOO/WBIC use the same chain as the saved .gz file (which is already filtered).
    """
    if os.path.exists(h5_path):
        r = emcee.backends.HDFBackend(h5_path, read_only=True)
        n_iter = int(r.iteration)
        if n_iter > 0:
            discard = int(DISCARD_FRAC * n_iter)
            chain3d = r.get_chain(discard=discard)          # (steps, walkers, ndim)
            lp2d = r.get_log_prob(discard=discard)          # (steps, walkers)
            if drop_stuck_walkers:
                chain3d, lp2d, _stuck = drop_stuck(chain3d, lp2d)
            n_kept = chain3d.shape[0] * chain3d.shape[1]
            thin = max(1, n_kept // THIN_TARGET)
            samples = chain3d[::thin].reshape(-1, chain3d.shape[2])
            log_lik = lp2d[::thin].reshape(-1)
            return samples, log_lik

    if not os.path.exists(gz_path):
        return None, None
    samples = np.loadtxt(gz_path)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.shape[0] > THIN_TARGET:
        idx = np.linspace(0, samples.shape[0] - 1, THIN_TARGET).astype(int)
        samples = samples[idx]
    log_lik = np.array([
        log_likelihood(t, model_extension, full_data, group, mouse_index)
        for t in samples
    ])
    return samples, log_lik


def _log_lik_at(theta, model_extension, full_data, group, mouse_index=None):
    return log_likelihood(theta, model_extension, full_data, group, mouse_index)


# === Comparison ===

def _add_weights_and_deltas(df, score_col):
    """Delta and weight of `score_col` within each (group, mouse)."""
    dcol, wcol = f'd{score_col}', f'w_{score_col}'
    df[dcol] = np.nan
    df[wcol] = np.nan
    for _key, sub in df.groupby(['group', 'mouse'], sort=False):
        s = sub[score_col].to_numpy(dtype=float)
        if not np.isfinite(s).any():
            continue
        df.loc[sub.index, dcol] = s - np.nanmin(s)
        df.loc[sub.index, wcol] = bic_weights(s)
    return df


ALL_METRICS = ('aic', 'aicc', 'bic', 'dic', 'waic', 'loo', 'wbic')
# Metrics that need only max_ll (the chain is not read)
_METRICS_MAX_LL_ONLY = ('aic', 'aicc', 'bic')
# Metrics that need the pointwise likelihood (one ODE solution per sample)
_METRICS_POINTWISE = ('waic', 'loo')
# Metric -> (score column, auxiliary columns)
_METRIC_COLS = {'aic': ('AIC', []),
                'aicc': ('AICc', []),
                'bic': ('BIC', []),
                'dic': ('DIC', ['p_D']),
                'waic': ('WAIC', ['p_waic', 'waic_high_var_points']),
                'loo': ('LOO', ['p_loo', 'khat_max', 'khat_bad_frac']),
                'wbic': ('WBIC', ['WBIC_ESS'])}


def normalize_metrics(metrics):
    """Validates and orders the selected metrics; None -> all."""
    if metrics is None:
        return list(ALL_METRICS)
    if isinstance(metrics, str):
        metrics = metrics.split(',')
    selected = [m.strip().lower() for m in metrics if str(m).strip()]
    unknown = [m for m in selected if m not in ALL_METRICS]
    if unknown:
        raise ValueError(f'Unknown metric(s): {unknown}. '
                         f'Use a subset of {list(ALL_METRICS)}.')
    if not selected:
        raise ValueError('Select at least one metric.')
    # canonical order, without duplicates
    return [m for m in ALL_METRICS if m in selected]


def compare_models(results_dir, full_data, group, mode='global',
                   models: Iterable = DEFAULT_MODELS, mouse_index=None,
                   metrics=None, n_pointwise=2000):
    """One row per model with the raw scores (no deltas or weights).

    `metrics` is a subset of ALL_METRICS. The chain is read only for metrics other than aic/aicc/bic.
    """
    metrics = normalize_metrics(metrics)
    needs_chain = any(m not in _METRICS_MAX_LL_ONLY for m in metrics)

    mouse = (full_data[group][mouse_index]['name']
             if mouse_index is not None else None)
    n_obs = n_observations(full_data, group, mouse_index)

    rows = []
    for model in models:
        npz_path, gz_path, h5_path = _paths(results_dir, mode, group, model,
                                            mouse)
        max_ll, theta_best = load_max_ll(npz_path, h5_path)
        if max_ll is None:
            continue
        k = len(theta_best)

        row = dict(group=group, mouse=mouse or '-', model=model,
                   n_obs=n_obs, k=k, n_samples=0, max_ll=max_ll)
        if 'aic' in metrics:
            row['AIC'] = compute_aic(max_ll, k)
        if 'aicc' in metrics:
            row['AICc'] = compute_aicc(max_ll, k, n_obs)
        if 'bic' in metrics:
            row['BIC'] = compute_bic(max_ll, k, n_obs)
        if 'dic' in metrics:
            row['DIC'] = row['p_D'] = float('nan')
        if 'waic' in metrics:
            row['WAIC'] = row['p_waic'] = float('nan'); row['waic_high_var_points'] = 0
        if 'loo' in metrics:
            row['LOO'] = row['p_loo'] = float('nan')
            row['khat_max'] = row['khat_bad_frac'] = float('nan')
        if 'wbic' in metrics:
            row['WBIC'] = row['WBIC_ESS'] = float('nan')

        if needs_chain:
            samples, log_lik = load_chain_and_loglik(
                h5_path, gz_path, model, full_data, group, mouse_index)
            if samples is not None:
                row['n_samples'] = int(len(log_lik))
                if 'dic' in metrics:
                    ll_at_mean = _log_lik_at(samples.mean(axis=0), model,
                                             full_data, group, mouse_index)
                    row['DIC'], row['p_D'] = compute_dic(log_lik, ll_at_mean)
                if 'wbic' in metrics:
                    row['WBIC'], row['WBIC_ESS'] = compute_wbic(log_lik, n_obs)

                if any(m in metrics for m in _METRICS_POINTWISE):
                    llmat = pointwise_loglik(samples, model, full_data, group,
                                             mouse_index, n_sub=n_pointwise)
                    if llmat is not None:
                        if 'waic' in metrics:
                            (row['WAIC'], row['p_waic'],
                             row['waic_high_var_points']) = compute_waic(llmat)
                        if 'loo' in metrics:
                            (row['LOO'], row['p_loo'], row['khat_max'],
                             row['khat_bad_frac']) = compute_loo(llmat)

        rows.append(row)
    return rows


def _fmt(v, w, p=2):
    return f'{v:>{w}.{p}f}' if np.isfinite(v) else f'{"-":>{w}s}'


def print_comparison_table(df, metrics=None):
    """Prints, for each (group, mouse), the table sorted by the first selected metric."""
    if df.empty:
        print('[model_selection] nothing to compare')
        return
    metrics = normalize_metrics(metrics)
    scores = [_METRIC_COLS[m][0] for m in metrics]
    primary = scores[0]

    for (group, mouse), sub in df.groupby(['group', 'mouse'], sort=False):
        target = group if mouse == '-' else f'{group} [{mouse}]'
        sub = sub.sort_values(primary)
        print(f'\n=== Model selection: {target}  '
              f'(n={int(sub["n_obs"].iloc[0])}) ===')

        header = f'{"":2s}{"model":<9s}{"k":>3s}{"max_ll":>11s}'
        for s in scores:
            header += f'{s:>10s}{"d"+s:>8s}{"w_"+s:>8s}'
            if s == 'DIC':
                header += f'{"p_D":>10s}'
            if s == 'WAIC':
                header += f'{"p_waic":>9s}'
            if s == 'LOO':
                header += f'{"p_loo":>8s}{"khat":>7s}'
            if s == 'WBIC':
                header += f'{"ESS":>8s}'
        print(header)

        best_index = {s: sub[s].idxmin() for s in scores if sub[s].notna().any()}
        for i, (_idx, r) in enumerate(sub.iterrows()):
            line = (f'{"*" if i == 0 else " "} {r["model"]:<8s}'
                     f'{int(r["k"]):>3d}{r["max_ll"]:>11.2f}')
            for s in scores:
                line += (_fmt(r[s], 10) + _fmt(r[f'd{s}'], 8)
                          + _fmt(r[f'w_{s}'], 8, 3))
                if s == 'DIC':
                    flag = '!' if np.isfinite(r['p_D']) and r['p_D'] < 0 else ' '
                    line += _fmt(r['p_D'], 9) + flag
                if s == 'WAIC':
                    flag = '!' if r.get('waic_high_var_points', 0) > 0 else ' '
                    line += _fmt(r['p_waic'], 8, 1) + flag
                if s == 'LOO':
                    kh = r['khat_max']
                    flag = '!' if np.isfinite(kh) and kh > 0.7 else ' '
                    line += _fmt(r['p_loo'], 8, 1) + _fmt(kh, 6, 2) + flag
                if s == 'WBIC':
                    line += _fmt(r['WBIC_ESS'], 8, 0)
            print(line)

        winners = {s: sub.loc[i, 'model'] for s, i in best_index.items()}
        print('  best -> ' + '  '.join(f'{s}: {v}'
                                         for s, v in winners.items()))
        if len(set(winners.values())) == 1:
            print(f'  agreement: {next(iter(winners.values()))}')
        elif len(winners) > 1:
            print('  metrics disagree (BIC penalizes complexity more; WBIC is valid for '
                  'singular models)')
        if 'DIC' in scores and (sub['p_D'] < 0).any():
            print('  ! p_D < 0: non-Gaussian posterior, DIC unreliable')
        if 'WAIC' in scores and (sub['waic_high_var_points'] > 0).any():
            n_max = int(sub['waic_high_var_points'].max())
            print(f'  ! WAIC: up to {n_max} observation(s) with Var(log p) > 0.4; '
                  'the estimate may be biased (prefer LOO)')
        if 'LOO' in scores and (sub['khat_max'] > 0.7).any():
            print('  ! khat > 0.7: heavy-tailed PSIS weights; LOO is unreliable for '
                  'those observations')
        if 'WBIC' in scores and (sub['WBIC_ESS'] < 50).any():
            print('  ! low WBIC_ESS: concentrated importance weights, '
                  'noisy WBIC')
        if 'AIC' in scores and 'AICc' not in scores:
            ratio = sub['n_obs'].iloc[0] / sub['k'].max()
            if ratio < 40:
                print(f'  ! n/k = {ratio:.1f} < 40: prefer AICc to AIC '
                      '(Burnham & Anderson)')


def run_model_selection(results_dir='.', mode='global',
                        groups: Iterable = DEFAULT_GROUPS,
                        models: Iterable = DEFAULT_MODELS,
                        full_data=None, metrics=None, n_pointwise=2000,
                        save_csv=True, print_table=True) -> pd.DataFrame:
    """Compares the models within each group (and each mouse in the mouse-specific mode).

    `metrics` is a subset of ALL_METRICS (None -> all). Writes
    {results_dir}/Output_Calibration/model_selection_{mode}.csv.
    """
    metrics = normalize_metrics(metrics)
    if full_data is None:
        from data_loader import load_control_data
        full_data = load_control_data()

    all_rows = []
    for group in groups:
        targets = ([None] if mode == 'global'
                 else list(range(len(full_data[group]))))
        for mouse_index in targets:
            all_rows.extend(compare_models(results_dir, full_data, group, mode,
                                        models, mouse_index, metrics,
                                        n_pointwise))

    if not all_rows:
        if print_table:
            print(f'[model_selection] no results in {results_dir}')
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    cols = ['group', 'mouse', 'model', 'n_obs', 'k', 'n_samples', 'max_ll']
    for m in metrics:
        score, aux = _METRIC_COLS[m]
        df = _add_weights_and_deltas(df, score)
        cols += [score, f'd{score}', f'w_{score}'] + aux
    df = df[[c for c in cols if c in df.columns]]

    if print_table:
        print_comparison_table(df, metrics)

    if save_csv:
        out_dir = os.path.join(results_dir, 'Output_Calibration')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'model_selection_{mode}.csv')
        df.to_csv(path, index=False, float_format='%.4f')
        if print_table:
            print(f'\nCSV: {path}')

    return df


def _cli():
    ap = argparse.ArgumentParser(
        description='Model selection: AIC, AICc, BIC, DIC, WAIC, LOO and WBIC with deltas and weights.')
    ap.add_argument('results_dir', nargs='?', default='.')
    ap.add_argument('--mode', default='global',
                    choices=['global', 'mouse_specific'])
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS))
    ap.add_argument('--groups', default=','.join(DEFAULT_GROUPS))
    ap.add_argument('--metrics', default=','.join(ALL_METRICS),
                    help="metrics to compute, a subset of "
                         "'aic,aicc,bic,dic,waic,loo,wbic' (default: all), e.g. --metrics bic")
    ap.add_argument('--no-dic', action='store_true',
                    help='shortcut for --metrics bic (chains are not read)')
    ap.add_argument('--no-csv', action='store_true')
    args = ap.parse_args()

    metrics = ['bic'] if args.no_dic else args.metrics

    run_model_selection(
        results_dir=args.results_dir, mode=args.mode,
        groups=args.groups.split(','), models=args.models.split(','),
        metrics=metrics, save_csv=not args.no_csv, print_table=True,
    )


if __name__ == '__main__':
    _cli()
