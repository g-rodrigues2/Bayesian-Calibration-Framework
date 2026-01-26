import os
import pandas as pd
import glob
import math
import emcee
import string
import corner
import numpy as np
from tqdm import tqdm
from numba import jit
import concurrent.futures
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from scipy.integrate import odeint
import seaborn as sns
import scipy.stats as stats


data_files = sorted(glob.glob("./data/*.txt"))
full_data = {}

for file in data_files:
    data_type = os.path.basename(file).split('/')[-1].split('_c')[0]
    mouse_name = 'c' + file.split('/')[-1].split('_c')[-1].split('_t')[0]
    t_days = np.array([int(t) for t in file.split('/')[-1].split('_c')[-1].split('.txt')[0].split('_t')[1:]])
    data = np.loadtxt(file)
    treatment = os.path.basename(data_type).split('_')[0]

    if data_type not in full_data:
        full_data[data_type] = []

    full_data[data_type].append({
        'name': mouse_name,
        'data': data,
        'treatment': treatment,
        'treatment_days': t_days
    })

mouse_name = full_data['control_resistant'][0]
data = mouse_name['data']


def ccc(x, y):
    """
    Calculate the Concordance Correlation Coefficient (CCC).

    Parameters:
    - x, y: Input data arrays

    Returns:
    - CCC value
    """
    x, y = np.asarray(x), np.asarray(y)

    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must not be empty")

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)

    if var_x == 0 or var_y == 0:
        raise ValueError("Input arrays must not have zero variance")

    covariance = np.cov(x, y, bias=True)[0, 1]
    ccc_value = (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)

    return ccc_value


def pcc(x, y):
    """
    Calculate the Pearson Correlation Coefficient (PCC).

    Parameters:
    - x, y: Input data arrays

    Returns:
    - PCC value
    """
    x, y = np.asarray(x), np.asarray(y)

    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must not be empty")

    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x), np.std(y)

    if std_x == 0 or std_y == 0:
        raise ValueError("Input arrays must not have zero variance")

    covariance = np.cov(x, y, bias=True)[0, 1]
    pcc_value = covariance / (std_x * std_y)

    return pcc_value


def nrmse(actual, pred):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE).

    Parameters:
    - actual: Actual values
    - pred: Predicted values

    Returns:
    - NRMSE value
    """
    actual, pred = np.asarray(actual), np.asarray(pred)

    if len(actual) != len(pred):
        raise ValueError("Input arrays must have the same length")

    if actual.size == 0 or pred.size == 0:
        raise ValueError("Input arrays must not be empty")

    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    nrmse_value = rmse / np.mean(actual) * 100

    return nrmse_value


def mape(pred, actual):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters:
    - actual: Actual values
    - pred: Predicted values

    Returns:
    - MAPE value
    """
    actual, pred = np.asarray(actual), np.asarray(pred)

    if len(actual) != len(pred):
        raise ValueError("Input arrays must have the same length")

    if actual.size == 0 or pred.size == 0:
        raise ValueError("Input arrays must not be empty")

    if np.any(actual == 0):
        raise ValueError("Actual values must not contain zeros")

    mape_value = np.mean(np.abs((actual - pred) / actual)) * 100

    return mape_value


@jit(nopython=True)
def model_exp(y, t, growth_rate):
    tumor = y
    tumorVolume = growth_rate * tumor
    return tumorVolume


@jit(nopython=True)
def model_bert(y, t, growth_rate, b):
    tumor = y
    tumorVolume = growth_rate * (tumor ** (2. / 3.)) - b * tumor
    return tumorVolume


@jit(nopython=True)
def model_mendel(y, t, growth_rate, power):
    tumor = y
    tumorVolume = growth_rate * (tumor ** power)
    return tumorVolume


@jit(nopython=True)
def model_log(y, t, growth_rate, carrying_capacity):
    tumor = y
    tumorVolume = growth_rate * tumor * (1 - tumor / carrying_capacity)
    return tumorVolume


@jit(nopython=True)
def model_lin(y, t, growth_rate, carrying_capacity):
    tumor = y
    tumorVolume = growth_rate * tumor / (tumor + carrying_capacity)
    return tumorVolume


@jit(nopython=True)
def model_gomp(y, t, growth_rate, b, c):
    tumor = y
    tumorVolume = growth_rate * tumor * np.log(b / (tumor + c))
    return tumorVolume


@jit(nopython=True)
def model_surf(y, t, growth_rate, carrying_capacity):
    tumor = y
    tumorVolume = growth_rate * tumor / ((tumor + carrying_capacity) ** (1. / 3.))
    return tumorVolume


def solve_model(model_extension, time_array, parameters, initial_condition, type_sol='data'):
    model_name = 'model' + model_extension
    model_func = globals()[model_name]
    rhs = getattr(model_func, 'py_func', model_func)
    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p - bgn_p) / 0.1) + 1)
        sol = odeint(rhs, t=time_array, y0=[initial_condition], args=parameters, mxstep=2000)
        return np.column_stack((time_array, sol))
    else:
        return odeint(rhs, t=time_array, y0=[initial_condition], args=parameters, mxstep=2000)


def log_likelihood(theta, model_extension, full_data, group, mouse_index):
    """
    Compute the likelihood for the specified model and dataset.

    Parameters:
    - theta: Model parameters.
    - model_extension: Model to be used (e.g., '_exp', '_log').
    - full_data: Dictionary containing group and mouse data.
    - group: Group name (e.g., 'control_resistant').
    - mouse_index: Index of the mouse within the group list.

    Returns:
    - ll: Log-likelihood value.
    """
    ll = 0
    variance = theta[-1] ** 2
    mouse_data = full_data[group][mouse_index]
    time_array = mouse_data['data'][:, 0]
    parameters = tuple(theta[:-2])
    initial_condition = theta[-2]
    solution = solve_model(model_extension, time_array, parameters, initial_condition)
    observed_volume = mouse_data['data'][:, 1]
    ll += -0.5 * np.sum(
        (solution[:, 0] - observed_volume) ** 2 / variance
        + np.log(2 * np.pi)
        + np.log(variance)
    )
    return ll


def log_prior(theta, l_bound, u_bound):
    """
    Logarithm of the prior probability for the model parameters.

    Parameters:
    - theta: Model parameters
    - l_bound: Lower bounds for the parameters
    - u_bound: Upper bounds for the parameters

    Returns:
    - Logarithm of the prior probability
    """
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0


def log_probability(theta, l_bound, u_bound, model_extension, full_data, group, mouse_index):
    """
    Compute the total log-probability (likelihood + prior) for the specified model and dataset.

    Parameters:
    - theta: Model parameters.
    - l_bound: Lower bounds for the parameters.
    - u_bound: Upper bounds for the parameters.
    - model_extension: Model to be used (e.g., '_exp', '_log').
    - full_data: Dictionary containing group and mouse data.
    - group: Group name.
    - mouse_index: Index of the mouse within the group list.

    Returns:
    - Total log-probability.
    """
    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model_extension, full_data, group, mouse_index)


def color_plot_scatter(final_exp, final_model, save, show, figure_name, formatter):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig, ax = plt.subplots(figsize=(5, 5))
    c = 1
    all_cccs = []
    for i in range(len(final_exp)):
        all_cccs.append(ccc(final_model[i], final_exp[i]))
        if all_cccs[-1] < 0.8:
            ax.plot(final_exp[i], final_model[i], 'o', color=colors[c % len(colors)])
            c += 1
        else:
            ax.plot(final_exp[i], final_model[i], 'o', color='black')
    ax.set_xlabel('Data - Tumor volume (mm³)')
    ax.set_ylabel('Model - Tumor volume (mm³)')
    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)
    final_exp = np.concatenate(final_exp)
    final_model = np.concatenate(final_model)
    max_value = max((max(final_exp), max(final_model)))
    max_value = math.ceil(max_value / 1000) * 1000
    ticks = np.linspace(0, max_value, num=int(max_value / 1000) + 1)

    ax.set_ylim((0, max_value))
    ax.set_xlim((0, max_value))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

    pccT = pcc(final_model, final_exp)
    cccT = ccc(final_model, final_exp)
    mapeT = mape(final_model, final_exp)
    all_cccs = np.array(all_cccs)
    ax.text(
        0.025, .8,
        'CCC/PCC = {:.2f}/{:.2f}\nMAPE = {:.2f}%\n CCC = {:.2f}$\\pm${:.2f}'.format(
            cccT, pccT, mapeT, all_cccs.mean(), all_cccs.std()
        ),
        horizontalalignment='left', transform=ax.transAxes
    )

    if save:
        plt.savefig(figure_name + '_sp.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close()
    return


def find_max_time(full_data):
    max_times = {}
    for group, mice_data in full_data.items():
        max_time = 0
        for mouse in mice_data:
            time_data = mouse['data'][:, 0]
            max_time_mouse = max(time_data)
            if max_time_mouse > max_time:
                max_time = max_time_mouse
        max_times[group] = max_time
    return max_times


max_times = find_max_time(full_data)


def configure_plot_settings(fontsize):
    plt.rcParams['font.size'] = fontsize
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))
    return formatter


def finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, figure_name, formatter):
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


def plot_maxll_solution(files_location, full_data, nCols=3, show=False, save=True, fontsize='14', figure_name='maxll_figure'):
    all_chains = sorted(glob.glob(files_location))
    plt.rcParams['font.size'] = fontsize

    if not show and not save:
        return

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))

    nScenarios = len(all_chains)
    nRows = int(np.ceil(nScenarios / nCols))
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 11, nRows * 5))
    axes = axes.ravel()

    full_exp = []
    full_model = []

    for idx, chain in enumerate(all_chains):
        ax = axes[idx]

        chain_split = chain.split('_')
        group = f"{chain_split[5]}_{chain_split[6]}"
        mouse_name = f"{chain_split[7]}_{chain_split[8]}"
        model_extension = f"_{chain_split[-1].split('.')[0]}"

        if group in full_data:
            max_times = find_max_time(full_data)
            max_times = max_times[group]
            mouse_data = next((m for m in full_data[group] if m['name'] == mouse_name), None)
            if mouse_data is None:
                print(f"Mouse {mouse_name} not found in group {group}")
                continue
        else:
            print(f"Group {group} not found in full_data")
            continue

        data = mouse_data['data']
        ax.set_ylabel('Tumor Volume (mm³)')

        ax.text(
            0.05, .9, '{:}) {:}'.format(string.ascii_uppercase[idx], mouse_name),
            horizontalalignment='left', transform=ax.transAxes, fontsize=18
        )

        ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10, label='Tumor Volume')
        ax.set_xlabel('Time (days)')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlim((-1, max_times + 1))

        npzfile = np.load(chain)
        theta = npzfile['pars']

        parameters = tuple(theta[:-2])
        initial_condition = theta[-2]

        solution = solve_model(model_extension, data[:, 0], parameters, initial_condition, "smooth")

        mask = np.isin(solution[:, 0], data[:, 0])
        matched_solution_times = solution[mask, 0]
        matched_solution_volumes = solution[mask, 1]

        ax.plot(
            matched_solution_times, matched_solution_volumes,
            color='black', linestyle='dashed', linewidth=2, label='Model'
        )

        full_model.append(matched_solution_volumes)
        full_exp.append(data[:, 1])

        pccT = pcc(matched_solution_volumes, data[:, 1])
        cccT = ccc(matched_solution_volumes, data[:, 1])
        mapeT = mape(matched_solution_volumes, data[:, 1])

        ax.text(
            0.5, .05,
            'CCC/PCC/MAPE = {:.2f}/{:.2f}/{:.2f}%'.format(cccT, pccT, mapeT),
            horizontalalignment='left', transform=ax.transAxes
        )

    finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, figure_name, formatter)
    color_plot_scatter(full_exp, full_model, save, show, figure_name, formatter)
    return


def find_max_initial_condition(full_data):
    """
    Find the largest initial condition (IC) among all mice across all groups.

    Parameters:
    - full_data: Dictionary containing all mice data grouped by treatment.

    Returns:
    - max_ic: The maximum IC value found.
    - max_ic_group: The group to which the mouse with the maximum IC belongs.
    - max_ic_mouse: The mouse name associated with the maximum IC.
    """
    max_ic = -float('inf')
    max_ic_group = None
    max_ic_mouse = None

    for group, mice_data in full_data.items():
        for mouse in mice_data:
            ic = mouse['data'][0, 1]
            if ic > max_ic:
                max_ic = ic
                max_ic_group = group
                max_ic_mouse = mouse['name']

    return max_ic, max_ic_group, max_ic_mouse


max_ic = find_max_initial_condition(full_data)
max_ic = max_ic[0]


def define_bounds_labels(model_extension, bool_multiple_mice=False):
    labels = []
    l_bound = []
    u_bound = []
    ic_max = max_ic * 2

    if model_extension == '_exp':
        labels = ["r"]
        l_bound = [0.0]
        u_bound = [1.0]
    elif model_extension == '_mendel':
        labels = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [1.0, 1.0]
    elif model_extension == '_bert':
        labels = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [2.0, 1.0]
    elif model_extension == '_lin':
        labels = ["r", "b"]
        l_bound = [0.0, 0.0000]
        u_bound = [1.0, ic_max]
    elif model_extension == '_gomp':
        labels = ["r", "b", "c"]
        l_bound = [1e-6, 1e-6, 1e-6]
        u_bound = [1.0, 10 * ic_max, ic_max]
    elif model_extension == '_surf':
        labels = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [1.0, ic_max]
    elif model_extension == '_log':
        labels = ["r", "cc"]
        l_bound = [0.0, 100.]
        u_bound = [1.0, 20000]

    if bool_multiple_mice:
        for i in range(group_size):
            labels.append("ic" + str(i + 1))
            l_bound.append(0.0000)
            u_bound.append(ic_max)
        labels.append("std")
        l_bound.append(0.1)
        u_bound.append(1000.0)
    else:
        labels.append("ic")
        l_bound.append(0.0000)
        u_bound.append(ic_max)
        labels.append("std")
        l_bound.append(0.1)
        u_bound.append(1000.0)

    return l_bound, u_bound, labels


run_calibration = True
if run_calibration:
    for model_extension in ['_exp', '_mendel', '_bert', '_lin', '_gomp', '_surf', '_log']:
        l_bound, u_bound, labels = define_bounds_labels(model_extension, bool_multiple_mice=False)
        ndim = len(l_bound)
        chain_size = 10
        nwalkers = 2 * ndim

        pos = np.zeros((nwalkers, ndim))
        for i in range(ndim):
            pos[:, i] = np.random.uniform(low=l_bound[i], high=u_bound[i], size=nwalkers)

        for group in ['control_sensitive', 'control_resistant']:
            print(f'Calibrating {group}, using model {model_extension.split("_")[-1]}')
            for mouse_index, mouse in enumerate(full_data[group]):
                mouse_name = mouse['name']

                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability,
                    args=(l_bound, u_bound, model_extension, full_data, group, mouse_index)
                )
                out_cal = sampler.run_mcmc(pos, chain_size, progress=True)

                flat_ll = sampler.get_log_prob(flat=True)
                flat_chain = sampler.get_chain(flat=True)
                max_pos = flat_ll.argmax(axis=0)
                best_pars = flat_chain[max_pos]
                size_discard = chain_size - 10000 // nwalkers
                save_chain = sampler.get_chain(discard=size_discard, flat=True)

                np.savetxt(f'./Output_Calibration_per_mouse/chain_{group}_{mouse_name}{model_extension}.gz', save_chain)
                np.savez(
                    f'./Output_Calibration_per_mouse/ll_pars_{group}_{mouse_name}{model_extension}.npz',
                    max_ll=max(flat_ll),
                    pars=best_pars
                )

                corner_chain = sampler.get_chain(discard=chain_size // 2, flat=True)
                fig = corner.corner(corner_chain, labels=labels)
                plt.savefig(
                    './Output_Calibration_per_mouse/corner_' + group + '_' + mouse_name + model_extension + '.png',
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0.02
                )
                plt.close()

                fig, axes = plt.subplots(nrows=len(labels), ncols=1, figsize=(10, len(labels) * 2), sharex=True)
                axes = axes.ravel()
                samples = sampler.get_chain()
                for i in range(len(labels)):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")
                plt.savefig(
                    './Output_Calibration_per_mouse/chain_' + group + '_' + mouse_name + model_extension + '.png',
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0.02
                )
                plt.close()

            plot_maxll_solution(
                './Output_Calibration_per_mouse/ll_pars_' + group + '_*' + model_extension + '.npz',
                full_data, nCols=4,
                figure_name='./Output_Calibration_per_mouse/max_ll_' + group + model_extension
            )
