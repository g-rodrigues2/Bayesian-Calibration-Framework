"""Post-processing: fit plots, posterior summaries and HDF5 compression.

Main functions:
  - compress_h5_inplace          : gzip compression of the datasets of an HDF5 backend
  - summarize_posterior          : median, 95% credible interval and 95% HDI (arviz optional)
  - posterior_corr_from_chain    : CCC/PCC with 95% credible intervals from 500 posterior samples
  - plot_scatter_ax              : observed versus predicted scatter panel with CCC/PCC
  - plot_maxll_solution_global   : fit of each mouse (global mode)
  - plot_maxll_solution_per_mouse: fit of each mouse (mouse-specific mode)
  - plot_maxll_solution          : dispatches to one of the two above
"""
import glob
import math
import os
import string

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from scipy.special import ndtr

from metrics import ccc, pcc
from models import solve_model, _split_theta


# Control groups shown side by side in the combined scatter plot
GROUP_LEFT  = 'control_sensitive'
GROUP_RIGHT = 'control_resistant'


class ScalarFormatterWithDecimals(ScalarFormatter):
    def _set_format(self):
        self.format = '%.1f'


def is_model_param(label):
    """True for a model parameter (not a per-mouse initial condition nor std)."""
    s = str(label).lower()
    return not (s.startswith('std') or s.startswith('ic'))


# === HDF5 utilities ===

def compress_h5_inplace(path, complevel=4, min_size=1024):
    """Rewrites the HDF5 file with gzip compression of the datasets larger than min_size.

    The gzip filter is native to HDF5, so emcee keeps reading and writing the backend transparently.
    maxshape is preserved so that the backend remains resumable. Typical reduction: 3-8x.
    """
    import h5py
    tmp = path + ".tmp"
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        for k in src.attrs:
            dst.attrs[k] = src.attrs[k]

        def copy_group(srcg, dstg):
            for k in srcg.attrs:
                dstg.attrs[k] = srcg.attrs[k]
            for name, item in srcg.items():
                if isinstance(item, h5py.Group):
                    copy_group(item, dstg.create_group(name))
                else:
                    data = item[...]
                    kwargs = dict(data=data, maxshape=item.maxshape)
                    if data.nbytes > min_size:
                        kwargs.update(compression='gzip',
                                      compression_opts=complevel,
                                      chunks=True)
                    ds = dstg.create_dataset(name, **kwargs)
                    for k in item.attrs:
                        ds.attrs[k] = item.attrs[k]

        copy_group(src, dst)
    os.replace(tmp, path)


# === Posterior summaries ===

def summarize_posterior(arr, alpha=0.05):
    """(median, lo, hi, hdi_lo, hdi_hi). The HDI comes from arviz when it is installed and imports
    cleanly; otherwise the percentile interval is returned."""
    arr = np.asarray(arr, dtype=float)
    med = np.median(arr)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    try:
        import arviz as az
        hdi_lo, hdi_hi = az.hdi(arr, hdi_prob=1 - alpha)
    except Exception:          # absent, or installed against an incompatible matplotlib
        hdi_lo, hdi_hi = lo, hi
    return med, lo, hi, hdi_lo, hdi_hi


def posterior_corr_from_chain(chain_path, model_extension, full_data, group,
                              mouse_index=None, n_samp=500, seed=42):
    """Median and 95% credible interval of CCC and PCC over n_samp posterior samples.

    mouse_index=None -> chain of the global mode (initial conditions of all mice).
    mouse_index=i    -> chain of the mouse-specific mode (a single initial condition).
    """
    if not os.path.exists(chain_path):
        print(f"[ERROR] Chain not found: {chain_path}")
        return None
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    if group not in full_data:
        print(f"[ERROR] Group '{group}' not found in full_data.")
        return None

    if mouse_index is None:
        mice = full_data[group]
    else:
        mice = [full_data[group][mouse_index]]
    ndim = len(mice)

    n_samp = min(n_samp, chain.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(chain.shape[0], size=n_samp, replace=False)
    thetas = chain[idx, :]

    pcc_samples, ccc_samples = [], []
    for k, theta in enumerate(thetas):
        try:
            ode_args, ic_fn = _split_theta(theta, model_extension, ndim)
            full_model, full_exp = [], []
            for i, mouse in enumerate(mice):
                t = mouse['data'][:, 0]
                y = mouse['data'][:, 1]
                sol = solve_model(model_extension, t, ode_args, ic_fn(i),
                                  type_sol='data')
                sol = sol[:, 0] if sol.ndim == 2 else sol
                full_model.append(sol)
                full_exp.append(y)
            model_concat = np.concatenate(full_model)
            exp_concat = np.concatenate(full_exp)
            pcc_val = float(np.clip(pcc(model_concat, exp_concat), -1.0, 1.0))
            ccc_val = float(np.clip(ccc(model_concat, exp_concat), -1.0, 1.0))
            pcc_samples.append(pcc_val)
            ccc_samples.append(ccc_val)
        except Exception as e:
            print(f"[WARN] sample failure {k}: {e}")

    pcc_samples = np.asarray(pcc_samples, dtype=float)
    ccc_samples = np.asarray(ccc_samples, dtype=float)
    pcc_summary = summarize_posterior(pcc_samples, alpha=0.05)
    ccc_summary = summarize_posterior(ccc_samples, alpha=0.05)

    tag = group if mouse_index is None else f"{group}[{mice[0]['name']}]"
    print(f"\n[Posterior {tag} | {model_extension}]  n={len(pcc_samples)}")
    print(f"PCC: median={pcc_summary[0]:.3f}, 95% CrI=[{pcc_summary[1]:.3f}, {pcc_summary[2]:.3f}], HDI95%=[{pcc_summary[3]:.3f}, {pcc_summary[4]:.3f}]")
    print(f"CCC: median={ccc_summary[0]:.3f}, 95% CrI=[{ccc_summary[1]:.3f}, {ccc_summary[2]:.3f}], HDI95%=[{ccc_summary[3]:.3f}, {ccc_summary[4]:.3f}]")

    return {
        "pcc_samples": pcc_samples, "ccc_samples": ccc_samples,
        "pcc_summary": pcc_summary, "ccc_summary": ccc_summary,
    }


def parameter_summary(chain_path, labels, params_of_interest=None,
                      print_table=False, title=''):
    """{label: (median, lo, hi, hdi_lo, hdi_hi)} of the posterior.

    params_of_interest=None -> all model parameters (initial conditions and std excluded).
    """
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    labels = [str(l) for l in labels]

    if params_of_interest is None:
        params_of_interest = [l for l in labels if is_model_param(l)]

    out = {}
    for name in params_of_interest:
        if name not in labels:
            continue
        idx = labels.index(name)
        out[name] = summarize_posterior(chain[:, idx], alpha=0.05)

    if print_table and out:
        print(f"\n[Parameters {title}]  median [95% CrI]  95% HDI")
        for name, (med, lo, hi, hlo, hhi) in out.items():
            print(f"  {name:<12s} {med:>12.5g}  [{lo:.5g}; {hi:.5g}]"
                  f"   HDI [{hlo:.5g}; {hhi:.5g}]")
    return out


def _mixture_quantile(f, sigma, q, max_iter=60):
    """Quantile q of the Gaussian mixture (1/n) * sum_i N(f_i(t), sigma_i^2).

    The posterior predictive distribution is this mixture. Inverting its CDF by bisection (vectorized
    over t) gives the exact band, without the Monte Carlo noise of sampling the observation error.

    f     : (n, T) curves f(t; theta_i)
    sigma : (n, 1) standard deviation of each sample
    """
    lo = (f - 10.0 * sigma).min(axis=0)
    hi = (f + 10.0 * sigma).max(axis=0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cdf = ndtr((mid[None, :] - f) / sigma).mean(axis=0)
        below = cdf < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def posterior_bands(chain, model_extension, time_array, ndim, mouse_idx,
                    n_samp=200, seed=42, alpha=0.05):
    """Posterior bands for one mouse, or None.

    n_samp parameter vectors are drawn from the chain and the ODE is solved for each on the 'smooth'
    grid (dt=0.1). Two bands are computed from the curves f(t; theta):

      'credible'   -- credible interval of the curve: percentiles of f(t; theta) over the posterior of
                      theta. It reflects parameter uncertainty only and is not expected to contain the
                      observations.

      'predictive' -- posterior predictive interval of a new observation: quantiles of
                      y ~ (1/n) sum_i N(f(t; theta_i), std_i^2), with std_i = theta_i[-1]. It adds the
                      observation noise and is expected to contain about 95% of the data (see
                      predictive_coverage).

    The predictive band is obtained by inverting the CDF of the Gaussian mixture (_mixture_quantile).

    Returns {'t', 'median', 'credible': (lo, hi), 'predictive': (lo, hi)}.
    """
    chain = np.asarray(chain)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    n_samp = min(n_samp, chain.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(chain.shape[0], size=n_samp, replace=False)

    curves, stds, t_grid = [], [], None
    for theta in chain[idx]:
        try:
            ode_args, ic_fn = _split_theta(theta, model_extension, ndim)
            sol = solve_model(model_extension, time_array, ode_args,
                              ic_fn(mouse_idx), type_sol='smooth')
        except Exception:
            continue
        if not np.all(np.isfinite(sol[:, 1])):
            continue
        t_grid = sol[:, 0]
        curves.append(sol[:, 1])
        stds.append(abs(float(theta[-1])))

    if not curves:
        return None

    f = np.vstack(curves)                       # (n, T)
    sigma = np.asarray(stds)[:, None]           # (n, 1)
    sigma = np.maximum(sigma, 1e-12)            # avoids division by zero
    qs = [100 * alpha / 2, 50, 100 * (1 - alpha / 2)]

    cred_lo, med, cred_hi = np.percentile(f, qs, axis=0)
    pred_lo = _mixture_quantile(f, sigma, alpha / 2)
    pred_hi = _mixture_quantile(f, sigma, 1 - alpha / 2)

    return {
        't': t_grid,
        'median': med,
        'credible': (cred_lo, cred_hi),
        'predictive': (pred_lo, pred_hi),
    }


def predictive_coverage(bands, mouse_data):
    """Fraction of the observations inside the 95% predictive band.

    Posterior predictive check: with a well-specified model and noise the expected value is about 0.95;
    much lower values indicate a misspecified model or std. Returns (fraction, n_inside, n_total), or
    None if there is no band.
    """
    if bands is None or 'predictive' not in bands:
        return None
    t_obs = mouse_data['data'][:, 0]
    y_obs = mouse_data['data'][:, 1]
    lo, hi = bands['predictive']
    # observed times on the smooth grid
    j = np.searchsorted(bands['t'], t_obs)
    j = np.clip(j, 0, len(bands['t']) - 1)
    inside = (y_obs >= lo[j]) & (y_obs <= hi[j])
    return float(inside.mean()), int(inside.sum()), int(len(y_obs))


# === Plots ===

def configure_plot_settings(fontsize):
    plt.rcParams['font.size'] = fontsize
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))
    return formatter


def find_max_time_per_group(full_data):
    return {grp: max(max(m['data'][:, 0]) for m in mice)
            for grp, mice in full_data.items() if mice}


def finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model,
                  save, show, figure_name, formatter):
    final_exp = np.concatenate(full_exp, axis=0) if full_exp else np.array([])
    final_model = np.concatenate(full_model, axis=0) if full_model else np.array([])
    if final_exp.size == 0 or final_model.size == 0:
        print("Error: One of the final arrays is empty.")
        return
    rounded_max = math.ceil(np.max(np.concatenate([final_exp, final_model])) / 1000) * 1000
    for ax in axes[:nScenarios]:
        ax.set_ylim((0, rounded_max))
        ax.yaxis.set_major_formatter(formatter)
    for i in range(nScenarios, nRows * nCols):
        fig.delaxes(axes[i])
    if save:
        plt.savefig(figure_name + '.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close()


def plot_scatter_ax(ax, final_exp, final_model, formatter,
                    ccc_val, ccc_ic, pcc_val, pcc_ic):
    for i in range(len(final_exp)):
        ax.plot(final_exp[i], final_model[i], 'o', color='blue')

    ax.set_xlabel(r'Data - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)
    ax.set_ylabel(r'Model - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)

    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)

    final_exp = np.concatenate(final_exp)
    final_model = np.concatenate(final_model)
    max_value = math.ceil(max(max(final_exp), max(final_model)) / 1000) * 1000
    ticks = np.linspace(0, max_value, num=int(max_value / 1000) + 1)
    ax.set_xlim((0, max_value)); ax.set_ylim((0, max_value))
    ax.set_xticks(ticks); ax.set_yticks(ticks)

    fmt = ScalarFormatterWithDecimals(useMathText=True)
    fmt.set_scientific(True); fmt.set_powerlimits((-3, 2))
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 2), useOffset=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', length=6, width=1,
                   direction='out', color='black', bottom=True, left=True)
    ax.minorticks_off()

    pccT = pcc(final_model, final_exp)
    cccT = ccc(final_model, final_exp)
    subplot_index = getattr(ax, 'get_subplotspec', lambda: None)()
    label_prefix = ''
    if subplot_index is not None:
        label_prefix = 'A) ' if subplot_index.colspan.start == 0 else 'B) '

    if pcc_ic is not None and ccc_ic is not None:
        ax.text(0.06, 0.90,
                f'{label_prefix}CCC = {ccc_val:.3f}  [{ccc_ic[0]:.3f}; {ccc_ic[1]:.3f}]',
                transform=ax.transAxes, fontsize=18, verticalalignment='top')
        ax.text(0.058, 0.835,
                f'     PCC = {pcc_val:.3f}  [{pcc_ic[0]:.3f}; {pcc_ic[1]:.3f}]',
                transform=ax.transAxes, fontsize=18, verticalalignment='top')
    else:
        ax.text(0.06, 0.9,
                f'{label_prefix}CCC/PCC = {cccT:.2f}/{pccT:.2f}',
                transform=ax.transAxes, fontsize=18, verticalalignment='top')


def _plot_mouse_panel(ax, mouse_data, model_extension, ode_args, ic,
                      idx, max_time_group, formatter, band=None,
                      central='median'):
    """Draws one panel (data and smooth solution) and returns (observed, maximum-likelihood model).

    band     - dict from posterior_bands (keys 't', 'median', 'credible', 'predictive'), or None.
    central  - 'median' (posterior median, consistent with the bands) or 'maxll' (maximum-likelihood
               curve). The maximum-likelihood curve is always dashed; `central` selects the solid line.

    The predictive band (light, outer) and the credible band (dark, inner) are drawn. The panel CCC/PCC
    and the returned values always use the maximum-likelihood curve.
    """
    data = mouse_data['data']
    ax.set_ylabel('Tumor Volume (mm³)')
    ax.text(0.05, .9, f'{string.ascii_uppercase[idx]}) {mouse_data["name"]}',
            horizontalalignment='left', transform=ax.transAxes, fontsize=18)
    ax.set_xlabel('Time (days)')
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlim((-1, max_time_group + 1))

    if band is not None:
        t_grid = band['t']
        if 'predictive' in band:
            plo, phi = band['predictive']
            ax.fill_between(t_grid, plo, phi, color='steelblue', alpha=0.13,
                            linewidth=0, label='95% predictive interval')
        if 'credible' in band:
            clo, chi = band['credible']
            ax.fill_between(t_grid, clo, chi, color='steelblue', alpha=0.35,
                            linewidth=0, label='95% credible band')
        if central == 'median':
            ax.plot(t_grid, band['median'], color='steelblue', linestyle='-',
                    linewidth=2, label='Model (median)')

    solution = solve_model(model_extension, data[:, 0], ode_args, ic, "smooth")
    mask = np.isin(solution[:, 0], data[:, 0])
    matched_v = solution[mask, 1]

    maxll_style = dict(color='black', linestyle='dashed', linewidth=2)
    if band is None or central == 'maxll':
        ax.plot(solution[:, 0], solution[:, 1], label='Model (max-LL)',
                **maxll_style)
    else:
        ax.plot(solution[:, 0], solution[:, 1], alpha=0.7,
                label='Model (max-LL)', **maxll_style)

    # data on top
    ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10,
            label='Tumor Volume', zorder=5)

    pccT = pcc(matched_v, data[:, 1])
    cccT = ccc(matched_v, data[:, 1])
    ax.text(0.5, .05, f'CCC/PCC = {cccT:.2f}/{pccT:.2f}',
            horizontalalignment='left', transform=ax.transAxes)
    if idx == 0:
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9,
                  bbox_to_anchor=(0.0, 0.86))
    return data[:, 1], matched_v


def _load_chain_for_band(chain_path):
    """Chain used for the bands, or None (with a warning) if it is missing."""
    if not os.path.exists(chain_path):
        print(f'[WARN] chain not found, bands disabled: {chain_path}')
        return None
    chain = np.loadtxt(chain_path)
    return chain.reshape(1, -1) if chain.ndim == 1 else chain


def _bands_for_mouse(chain, model_extension, mouse_data, ndim, mouse_idx,
                     n_band, kind='both', report_coverage=True, tag=''):
    """Bands of one mouse; optionally prints the predictive coverage."""
    if chain is None:
        return None
    bands = posterior_bands(chain, model_extension, mouse_data['data'][:, 0],
                            ndim, mouse_idx, n_samp=n_band)
    if bands is None:
        return None

    if report_coverage:
        cov = predictive_coverage(bands, mouse_data)
        if cov is not None:
            frac, k, n = cov
            print(f"  [95% predictive coverage] {tag}{mouse_data['name']}: "
                  f"{k}/{n} = {frac:.2%}")

    if kind == 'credible':
        bands.pop('predictive', None)
    elif kind == 'predictive':
        bands.pop('credible', None)
    return bands


def plot_maxll_solution_global(files_location, full_data, nCols=3, show=True,
                               save=True, fontsize='14', figure_name='maxll_figure',
                               out_dir='./Output_Calibration',
                               band=True, central='median', n_band=200,
                               band_kind='both'):
    """Fit of each mouse (with 95% credible and predictive bands) and combined scatter plot.

    Expects ``multi_ll_pars_{group}{ext}.npz`` files of the global mode. The bands come from the
    corresponding chain (``multi_chain_{group}{ext}.gz``); without it only the maximum-likelihood curve
    is drawn.
    """
    if not show and not save:
        return

    all_chains = sorted(glob.glob(files_location))
    formatter = configure_plot_settings(fontsize)

    best_chain_per_group = {}
    max_ll_per_group = {}
    for chain in all_chains:
        npzfile = np.load(chain)
        max_ll = npzfile['max_ll']
        stem = os.path.basename(chain).replace('multi_ll_pars_', '').rsplit('.', 1)[0]
        group, _model_token = stem.rsplit('_', 1)
        if group not in max_ll_per_group or max_ll > max_ll_per_group[group]:
            max_ll_per_group[group] = max_ll
            best_chain_per_group[group] = chain

    max_times = find_max_time_per_group(full_data)
    scatter_pairs = {}
    model_extension = None

    for group, best_chain in best_chain_per_group.items():
        print(f"Processing: Group = {group}, max_ll = {max_ll_per_group[group]}")
        npzfile = np.load(best_chain)
        theta = npzfile['pars']

        nScenarios = len(full_data[group])
        nRows = int(np.ceil(nScenarios / nCols))
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols,
                                 figsize=(nCols * 11, nRows * 5))
        axes = np.atleast_1d(axes).ravel()

        full_exp, full_model = [], []
        model_extension = '_' + os.path.basename(best_chain).rsplit('.', 1)[0].rsplit('_', 1)[1]
        ndim = len(full_data[group])
        print(f'Size pars = {len(theta)}, animals = {ndim}')
        max_time_group = max_times[group]
        ode_args, ic_fn = _split_theta(theta, model_extension, ndim)

        chain_path = f'{out_dir}/multi_chain_{group}{model_extension}.gz'
        chain = _load_chain_for_band(chain_path) if band else None

        for idx, mouse_data in enumerate(full_data[group]):
            bands = _bands_for_mouse(chain, model_extension, mouse_data, ndim,
                                     idx, n_band, kind=band_kind,
                                     tag=f'{group} ')
            exp_i, mod_i = _plot_mouse_panel(
                axes[idx], mouse_data, model_extension, ode_args, ic_fn(idx),
                idx, max_time_group, formatter, band=bands, central=central)
            full_exp.append(exp_i); full_model.append(mod_i)

        finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model,
                      save, show, figure_name, formatter)
        scatter_pairs[group] = (full_exp.copy(), full_model.copy())

    # Combined scatter plot of the two control groups
    if GROUP_LEFT in scatter_pairs and GROUP_RIGHT in scatter_pairs:
        (exp_s, mod_s) = scatter_pairs[GROUP_LEFT]
        (exp_r, mod_r) = scatter_pairs[GROUP_RIGHT]
        fig_sc, axes_sc = plt.subplots(1, 2, figsize=(22, 15 * (1.4 / 3.4)))

        res_cs = posterior_corr_from_chain(
            chain_path=f'{out_dir}/multi_chain_{GROUP_LEFT}{model_extension}.gz',
            model_extension=model_extension, full_data=full_data,
            group=GROUP_LEFT, n_samp=500, seed=42)
        res_cr = posterior_corr_from_chain(
            chain_path=f'{out_dir}/multi_chain_{GROUP_RIGHT}{model_extension}.gz',
            model_extension=model_extension, full_data=full_data,
            group=GROUP_RIGHT, n_samp=500, seed=42)

        if res_cs and res_cr:
            ccc_cs = res_cs["ccc_summary"]; pcc_cs = res_cs["pcc_summary"]
            ccc_cr = res_cr["ccc_summary"]; pcc_cr = res_cr["pcc_summary"]
            plot_scatter_ax(axes_sc[0], exp_s, mod_s, formatter,
                            ccc_val=ccc_cs[0], ccc_ic=[ccc_cs[1], ccc_cs[2]],
                            pcc_val=pcc_cs[0], pcc_ic=[pcc_cs[1], pcc_cs[2]])
            plot_scatter_ax(axes_sc[1], exp_r, mod_r, formatter,
                            ccc_val=ccc_cr[0], ccc_ic=[ccc_cr[1], ccc_cr[2]],
                            pcc_val=pcc_cr[0], pcc_ic=[pcc_cr[1], pcc_cr[2]])

            fig_sc.tight_layout()
            if save:
                fig_sc.savefig(f'{out_dir}/multi_scatter_both{model_extension}.png',
                               dpi=300, bbox_inches='tight', pad_inches=0.02)
                fig_sc.savefig(f'{out_dir}/multi_scatter_both{model_extension}.pdf',
                               bbox_inches='tight', pad_inches=0.02)
            if show:
                plt.show()
            else:
                plt.close(fig_sc)


def _per_mouse_entries(full_data, group, model_extension, out_dir):
    """[(index in full_data, mouse, npz_path, chain_path)] of the mice with an .npz file."""
    entries = []
    for idx, mouse in enumerate(full_data[group]):
        stem = f"{group}_{mouse['name']}{model_extension}"
        npz_path = f'{out_dir}/ll_pars_{stem}.npz'
        if os.path.exists(npz_path):
            entries.append((idx, mouse, npz_path, f'{out_dir}/chain_{stem}.gz'))
    return entries


def plot_maxll_solution_per_mouse(full_data, group, model_extension,
                                  out_dir='./Output_Calibration_per_mouse',
                                  nCols=3, show=True, save=True, fontsize='14',
                                  figure_name='maxll_figure_per_mouse',
                                  band=True, central='median', n_band=200,
                                  band_kind='both'):
    """Fit of each mouse of the group, each with the estimate and chain of its own calibration
    (mouse-specific mode).

    Expects ``ll_pars_{group}_{mouse}{ext}.npz`` and ``chain_{group}_{mouse}{ext}.gz``. Mice without
    an .npz file are skipped (e.g. a subset selected with CALIB_MICE).
    """
    if not show and not save:
        return

    formatter = configure_plot_settings(fontsize)
    max_time_group = find_max_time_per_group(full_data)[group]
    entries = _per_mouse_entries(full_data, group, model_extension, out_dir)

    if not entries:
        print(f"[WARN] No per-mouse .npz files in {out_dir} for "
              f"{group}{model_extension}")
        return

    nScenarios = len(entries)
    nRows = int(np.ceil(nScenarios / nCols))
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols,
                             figsize=(nCols * 11, nRows * 5))
    axes = np.atleast_1d(axes).ravel()

    full_exp, full_model = [], []
    for panel, (_idx, mouse_data, npz_path, chain_path) in enumerate(entries):
        theta = np.load(npz_path)['pars']
        # ndim=1: theta = [parameters..., ic, std]
        ode_args, ic_fn = _split_theta(theta, model_extension, 1)

        bands = None
        if band:
            chain = _load_chain_for_band(chain_path)
            bands = _bands_for_mouse(chain, model_extension, mouse_data, 1, 0,
                                     n_band, kind=band_kind, tag=f'{group} ')

        exp_i, mod_i = _plot_mouse_panel(
            axes[panel], mouse_data, model_extension, ode_args, ic_fn(0),
            panel, max_time_group, formatter, band=bands, central=central)
        full_exp.append(exp_i); full_model.append(mod_i)

    finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model,
                  save, show, figure_name, formatter)
    return full_exp, full_model


def scatter_both_per_mouse(full_data, groups, model_extension,
                           out_dir='./Output_Calibration_per_mouse',
                           figure_name=None, n_samp=500, seed=42,
                           show=False, save=True, fontsize='14'):
    """Side-by-side scatter plots of two groups in the mouse-specific mode.

    Pools the points of all mice of a group (each with its own maximum-likelihood estimate) and reports
    the median CCC/PCC with 95% credible intervals, from the concatenated posterior samples of the mice.
    """
    if len(groups) != 2:
        print(f'[scatter_both_per_mouse] requires exactly 2 groups, '
              f'got {groups}; skipping.')
        return
    formatter = configure_plot_settings(fontsize)
    figure_name = figure_name or f'{out_dir}/scatter_both_per_mouse{model_extension}'

    panels = []
    for group in groups:
        entries = _per_mouse_entries(full_data, group, model_extension, out_dir)
        if not entries:
            print(f'[scatter_both_per_mouse] no chains for {group}; skipping.')
            return

        exp_pts, mod_pts = [], []
        ccc_all, pcc_all = [], []
        for idx, mouse_data, npz_path, chain_path in entries:
            theta = np.load(npz_path)['pars']
            ode_args, ic_fn = _split_theta(theta, model_extension, 1)
            t = mouse_data['data'][:, 0]
            sol = solve_model(model_extension, t, ode_args, ic_fn(0))
            exp_pts.append(mouse_data['data'][:, 1])
            mod_pts.append(sol[:, 0])

            res = posterior_corr_from_chain(
                chain_path, model_extension, full_data, group,
                mouse_index=idx, n_samp=n_samp, seed=seed)
            if res is not None:
                ccc_all.append(res['ccc_samples'])
                pcc_all.append(res['pcc_samples'])

        # 95% credible interval over the pooled posterior samples of all mice
        ccc_s = summarize_posterior(np.concatenate(ccc_all)) if ccc_all else None
        pcc_s = summarize_posterior(np.concatenate(pcc_all)) if pcc_all else None
        panels.append((exp_pts, mod_pts, ccc_s, pcc_s))

    fig, axes = plt.subplots(1, 2, figsize=(22, 15 * (1.4 / 3.4)))
    for ax, (exp_pts, mod_pts, ccc_s, pcc_s) in zip(axes, panels):
        if ccc_s is None or pcc_s is None:
            plot_scatter_ax(ax, exp_pts, mod_pts, formatter,
                            None, None, None, None)
        else:
            plot_scatter_ax(ax, exp_pts, mod_pts, formatter,
                            ccc_val=ccc_s[0], ccc_ic=[ccc_s[1], ccc_s[2]],
                            pcc_val=pcc_s[0], pcc_ic=[pcc_s[1], pcc_s[2]])
    fig.tight_layout()
    if save:
        fig.savefig(figure_name + '.png', dpi=300, bbox_inches='tight',
                    pad_inches=0.02)
        fig.savefig(figure_name + '.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_maxll_solution(full_data, per_mouse=False, **kwargs):
    """Calls the global or the per-mouse variant."""
    if per_mouse:
        return plot_maxll_solution_per_mouse(full_data, **kwargs)
    return plot_maxll_solution_global(full_data=full_data, **kwargs)
