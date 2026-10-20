"""Plot style of the Chapter 4 figures, shared by the figure scripts of all stages.

The functions reproduce the layout of the figures of the thesis: Times New Roman text, tumor volume in
units of 10^3 mm^3, black observations, maximum-likelihood trajectory, light-blue 95% credible band and
peach 95% posterior predictive interval evaluated at the observation times, dashed treatment lines,
and the posterior median and 95% credible interval of CCC and PCC written in each panel. Histograms
of posterior samples use the default font, as in the thesis.
"""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BAND = 'lightblue'
PREDICTIVE = 'peachpuff'
HIST = '#4B88A2'
PASTEL = {'sensitive': '#a1c9f4', 'resistant': '#ffb482', 'global': '#8de5a1'}   # seaborn "pastel" 0, 1, 2
VOLUME_LABEL = r'Tumor Volume ($\times10^3$ mm³)'

# Posterior panels of Tables 4.5, 4.11, 4.14 and 4.18: drawn at the width they have on the page,
# with the text at the size it is printed.
PANEL_FONT = 9
PANEL_ASPECT = 0.72


class OneDecimal(mticker.ScalarFormatter):
    def _set_format(self):
        self.format = '%.1f'


def times_font(size=20):
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': size})


def default_font(size=14):
    plt.rcdefaults()
    plt.rcParams['font.size'] = size


def rounded_thousands(*arrays):
    return math.ceil(max(float(np.max(a)) for a in arrays) / 1000) * 1000


def volume_yaxis_fitted(ax, *arrays):
    """Volume axis scaled to the plotted values: top rounded up to 250 (below 1000 mm^3), 500 (below 4000)
    or 1000 mm^3 above the largest value plus 25% (room for the legend and labels), with tick labels in
    units of 10^3 mm^3."""
    peak = 1.25 * max(float(np.max(a)) for a in arrays)
    step = 250 if peak <= 1000 else 500 if peak <= 4000 else 1000
    top = math.ceil(peak / step) * step
    ax.set_ylim((0, top))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v / 1000:g}'))


def volume_yaxis(ax, top):
    """y axis from 0 to `top` mm^3, labelled in units of 10^3 mm^3."""
    ax.set_ylim((0, top))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(500 if top <= 4000 else 1000))
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_visible(False)


def _frame(ax, labelsize, length=6):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', length=length, width=1, direction='out', color='black',
                   bottom=True, left=True, labelsize=labelsize)
    ax.minorticks_off()
    ax.grid(False)


# Panel layouts of the thesis figures:
#   best_worst  Figures 4.6 and 4.9 (best and worst mouse of each phenotype, scatter plots below)
#   all_mice    Appendix Figures S1 and S2 (every mouse of a phenotype)
#   selected    Figures 4.11-4.13 and 4.14 (selected mice of the immunotherapy and combined cohorts)
#   all_mice_io Appendix figures of the combined cohort (every mouse)
LAYOUTS = {
    'best_worst': dict(ms=10, label=20, ticks=12, legend=16, tick_len=6, letter=(0.02, 0.90, 'baseline'),
                       ccc=(0.06, 0.96), pcc=(0.065, 0.865), gap='  '),
    'all_mice': dict(ms=6, label=20, ticks=13, legend=18, tick_len=6, letter=(0.02, 0.90, 'baseline'),
                     ccc=(0.06, 0.96), pcc=(0.065, 0.865), gap='  '),
    'selected': dict(ms=10, label=20, ticks=18, legend=16, tick_len=6, letter=(0.02, 0.92, 'top'),
                     ccc=(0.06, 0.92), pcc=(0.065, 0.835), gap=' '),
    'all_mice_io': dict(ms=6, label=16, ticks=13, legend=10, tick_len=5, letter=(0.02, 0.92, 'top'),
                        ccc=(0.06, 0.92), pcc=(0.065, 0.835), gap=' '),
}


def metrics_text(ax, layout, letter, ccc=None, pcc=None, fontsize=18):
    """Panel letter and posterior median [95% CrI] of CCC and PCC, at the positions of the thesis figures."""
    L = LAYOUTS[layout]
    x, y, va = L['letter']
    ax.text(x, y, f'{letter})', transform=ax.transAxes, fontsize=fontsize, va=va)
    if ccc is not None:
        g = L['gap']
        ax.text(*L['ccc'], f'CCC = {ccc[0]:.3f}{g}[{ccc[1]:.3f}; {ccc[2]:.3f}]', transform=ax.transAxes,
                fontsize=fontsize, va='top')
        ax.text(*L['pcc'], f'PCC = {pcc[0]:.3f}{g}[{pcc[1]:.3f}; {pcc[2]:.3f}]', transform=ax.transAxes,
                fontsize=fontsize, va='top')


def fit_panel(ax, data, curve, line_label, layout, *, band=None, predictive=None, extra_curves=(),
              treatments=(), xmax=None, line_color='black', legend=False):
    """One mouse: observations, trajectory `curve` = (t, T), optional bands (lower, upper) at the
    observation times, `extra_curves` = [(t, T, label, color, linestyle)], `treatments` = [(days, color)]."""
    L = LAYOUTS[layout]
    t, y = data[:, 0], data[:, 1]
    ax.plot(t, y, 'o', color='black', markersize=L['ms'], label='Tumor Volume')
    ax.plot(curve[0], curve[1], color=line_color, linestyle='solid', linewidth=2, label=line_label)
    for ct, cy, label, color, ls in extra_curves:
        ax.plot(ct, cy, color=color, linestyle=ls, linewidth=2, label=label)
    if predictive is not None:
        ax.fill_between(t, predictive[1], predictive[0], color=PREDICTIVE, alpha=0.45,
                        label='95% posterior predictive interval')
    if band is not None:
        ax.fill_between(t, band[1], band[0], color=BAND, alpha=0.8, label='95% credible band')
    for days, color in treatments:
        for day in days:
            ax.axvline(x=day, color=color, linestyle='--', linewidth=1)
    ax.set_xlabel('Time (days)', fontsize=L['label'])
    ax.set_ylabel(VOLUME_LABEL, fontsize=L['label'])
    if xmax is not None:
        ax.set_xlim((-1, xmax + 1))
    _frame(ax, L['ticks'], L['tick_len'])
    ax.xaxis.get_offset_text().set_visible(False)
    if legend:
        ax.legend(loc=1, fontsize=L['legend'])


def scatter_panel(ax, observed, predicted, letter, ccc, pcc, style='rtio', labelsize=None):
    """Observed versus predicted volumes (lists of arrays per mouse), identity line and posterior CCC/PCC.

    style: 'rtio' (Figures 4.11-4.14), 'best_worst' (Figures 4.6 and 4.9), 'inline' (Figures 4.3 and 4.8,
    panel letter written before the CCC)."""
    for o, p in zip(observed, predicted):
        ax.plot(o, p, 'o', color='blue')
    ax.set_xlabel(r'Data - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)
    ax.set_ylabel(r'Model - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)
    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)
    top = rounded_thousands(np.concatenate(observed), np.concatenate(predicted))
    ticks = np.linspace(0, top, num=int(top / 1000) + 1)
    ax.set_xlim((0, top))
    ax.set_ylim((0, top))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = OneDecimal(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-3, 2))
        axis.set_major_formatter(fmt)
        axis.get_offset_text().set_visible(False)
    if labelsize is None:
        labelsize = 12 if style == 'best_worst' else plt.rcParams['font.size']
    _frame(ax, labelsize)
    if style == 'inline':
        ax.text(0.06, 0.90, f'{letter}) CCC = {ccc[0]:.3f}  [{ccc[1]:.3f}; {ccc[2]:.3f}]',
                transform=ax.transAxes, fontsize=18, va='top')
        ax.text(0.058, 0.835, f'     PCC = {pcc[0]:.3f}  [{pcc[1]:.3f}; {pcc[2]:.3f}]',
                transform=ax.transAxes, fontsize=18, va='top')
        return
    if style == 'best_worst':
        ax.text(0.02, 0.9, f'{letter})', transform=ax.transAxes, fontsize=18)
    else:
        ax.text(0.02, 0.94, f'{letter})', transform=ax.transAxes, fontsize=18, va='top')
    ax.text(0.06, 0.94, f'CCC = {ccc[0]:.3f}  [{ccc[1]:.3f}; {ccc[2]:.3f}]', transform=ax.transAxes,
            fontsize=18, va='top')
    ax.text(0.065, 0.875, f'PCC = {pcc[0]:.3f}  [{pcc[1]:.3f}; {pcc[2]:.3f}]', transform=ax.transAxes,
            fontsize=18, va='top')


def _order(values):
    finite = np.abs(np.asarray(values, float))
    finite = finite[finite > 0]
    return int(math.floor(math.log10(finite.max()))) if finite.size else 0


def posterior_histogram(samples, best, path, width_on_page, label, bounds=None):
    """Posterior histogram with kernel density, drawn at the width it has on the printed page.

    `width_on_page` is that width in inches and `label` the parameter written below the axis, so that the
    text of the panel is printed at the 9 pt of the thesis. The dashed line marks the maximum-likelihood
    sample and the dotted lines the prior bounds, drawn only when they fall inside the plotted range.
    Tick labels are mantissas and the power of ten is written in the label."""
    import seaborn as sns

    times_font(PANEL_FONT)
    plt.rcParams['mathtext.fontset'] = 'cm'   # mathematics as in the body text
    fig, ax = plt.subplots(figsize=(width_on_page, width_on_page * PANEL_ASPECT))
    sns.histplot(np.asarray(samples, float), bins=30, kde=True, ax=ax, color=HIST, alpha=0.7, linewidth=0.3)
    if best is not None:
        ax.axvline(float(best), linestyle='--', linewidth=0.8, color='black')
    lo, hi = ax.get_xlim()
    for bound in bounds or ():
        if lo <= float(bound) <= hi:
            ax.axvline(float(bound), linestyle=':', linewidth=0.8, color='black')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    fig.canvas.draw()
    k = _order([v for v in ax.get_xticks() if lo <= v <= hi])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _, k=k: f'{v / 10 ** k:g}'))
    ax.set_xlabel(label + (rf' ($\times 10^{{{k}}}$)' if k else ''), fontsize=PANEL_FONT, labelpad=1.5)
    ax.tick_params(axis='x', labelsize=PANEL_FONT - 1, length=3, width=0.6, pad=1.5)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.grid(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def scenario_violins(ax, entries, ylabel, letter, width=0.3):
    """Posterior distributions of one parameter across sharing scenarios.

    entries: list over scenarios (x = 1, 2, ...) of either
        ('global', samples, ml_value) or ('group', (sensitive, resistant), (ml_sensitive, ml_resistant)).
    Each violin is a kernel density of the posterior samples; inside it, the thick bar is the interquartile
    range, the thin bar the 95% credible interval and the white dot the median; red stars mark the
    maximum-likelihood sample."""
    def one(x, samples, ml, color):
        samples = np.asarray(samples, float)
        parts = ax.violinplot([samples], positions=[x], widths=width, showextrema=False, showmedians=False)
        for body in parts['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor('black')
            body.set_linewidth(0.8)
            body.set_alpha(0.9)
        q025, q25, q50, q75, q975 = np.percentile(samples, [2.5, 25, 50, 75, 97.5])
        ax.vlines(x, q025, q975, color='black', linewidth=1.2, zorder=3)
        ax.vlines(x, q25, q75, color='black', linewidth=5, zorder=3)
        ax.scatter([x], [q50], s=28, color='white', edgecolor='black', linewidth=0.8, zorder=4)
        ax.plot(x, ml, 'r*', markersize=10, zorder=5)

    for scenario, (kind, samples, ml) in enumerate(entries, start=1):
        if kind == 'global':
            one(scenario, samples, ml, PASTEL['global'])
        else:
            one(scenario - 0.2, samples[0], ml[0], PASTEL['sensitive'])
            one(scenario + 0.2, samples[1], ml[1], PASTEL['resistant'])
    ax.set_xticks(range(1, len(entries) + 1))
    ax.set_xticklabels(range(1, len(entries) + 1), fontsize=18)
    ax.set_xlim(0.4, len(entries) + 0.6)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.text(0.02, 0.9, f'{letter})', transform=ax.transAxes, fontsize=20)


def violin_legend(ax, fontsize=16):
    handles = [mpatches.Patch(facecolor=PASTEL['sensitive'], edgecolor='black', label='Sensitive'),
               mpatches.Patch(facecolor=PASTEL['resistant'], edgecolor='black', label='Resistant'),
               mpatches.Patch(facecolor=PASTEL['global'], edgecolor='black', label='Global'),
               mlines.Line2D([], [], color='white', marker='o', markeredgecolor='black', linestyle='None',
                             markersize=7, label='Median'),
               mlines.Line2D([], [], color='black', linewidth=5, label='Interquartile range'),
               mlines.Line2D([], [], color='black', linewidth=1.2, label='95% credible interval'),
               mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=10,
                             label='Maximum likelihood')]
    ax.legend(handles=handles, loc=1, fontsize=fontsize, frameon=False, ncol=2)


def trace_figure(chain, labels, path, burn, removed=None, ncols=1, max_points=2000, title=None, step=1):
    """Walker traces (steps x walkers x parameters) of the selected parameters.

    Retained walkers are drawn in black, walkers removed as stuck in orange; the dashed red line marks the
    end of the burn-in period. `step` is the interval between stored samples, so that a thinned chain is
    drawn against the step numbers of the original one, as `burn` is given in those."""
    times_font(14)
    steps, walkers, n = chain.shape
    stride = max(1, steps // max_points)
    x = np.arange(0, steps, stride) * step
    removed = np.zeros(walkers, bool) if removed is None else np.asarray(removed, bool)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols / max(1, ncols ** 0.5), 1.9 * nrows),
                             sharex=True, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.remove()
            continue
        ax.plot(x, chain[::stride, ~removed, i], color='black', alpha=0.3, linewidth=0.4)
        if removed.any():
            ax.plot(x, chain[::stride, removed, i], color='darkorange', alpha=0.7, linewidth=0.5)
        ax.axvline(burn, color='red', linestyle='--', linewidth=1)
        ax.set_xlim(0, steps * step)
        ax.set_ylabel(labels[i], fontsize=14)
        ax.yaxis.set_label_coords(-0.1 if ncols == 1 else -0.18, 0.5)
        ax.tick_params(labelsize=11)
    for ax in axes[-1]:
        ax.set_xlabel('Step number', fontsize=14)
    if title:
        fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
