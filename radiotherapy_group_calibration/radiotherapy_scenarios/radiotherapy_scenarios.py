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
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from scipy.integrate import odeint
import seaborn as sns
import scipy.stats as stats
import sys
import argparse

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

# Concordance Correlation Coefficient
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

# Pearson Correlation Coefficient
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

# Normalized Root Mean Squared Error
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
    
    rmse = np.sqrt(np.mean((actual - pred)**2))
    nrmse_value = rmse / np.mean(actual) * 100
    
    return nrmse_value

# Mean Absolute Percentage Error
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
def model_ExponentialDecayDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, tau_radiation, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:
            treatment_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau - delay))

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

def solve_model(model_extension, time_array, parameters, initial_condition, treatment_days, dose, type_sol='data'):
    model_name = 'model' + model_extension
    model_func = globals()[model_name]

    tau_radiation = treatment_days
    dose_schedule = [dose] * len(tau_radiation)

    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p - bgn_p) / 0.1) + 1)
        sol = odeint(model_func, t=time_array, y0=[initial_condition], args=(*parameters, tau_radiation, dose_schedule), mxstep=2000)
        return np.column_stack((time_array, sol))
    else:
        return odeint(model_func, t=time_array, y0=[initial_condition], args=(*parameters, tau_radiation, dose_schedule), mxstep=2000)

def scenario_tag(scenario: int) -> str:
    return f"V{int(scenario)}"

def scenario_base_dim(scenario: int) -> int:
    # base params: alpha(s), beta(s), delta(s)
    return {1: 6, 2: 4, 3: 4, 4: 3, 5: 5, 6: 5, 7: 5, 8: 4}[int(scenario)]

def scenario_param_indices(scenario: int):
    # returns indices in theta for a_s,a_r,b_s,b_r,d_s,d_r (duplicating globals)
    m = {
        1: dict(a_s=0, a_r=1, b_s=2, b_r=3, d_s=4, d_r=5),                  # all group-specific
        2: dict(a_s=0, a_r=0, b_s=1, b_r=2, d_s=3, d_r=3),                  # a global, b gs, d global
        3: dict(a_s=0, a_r=1, b_s=2, b_r=2, d_s=3, d_r=3),                  # a gs, b global, d global
        4: dict(a_s=0, a_r=0, b_s=1, b_r=1, d_s=2, d_r=2),                  # all global
        5: dict(a_s=0, a_r=1, b_s=2, b_r=3, d_s=4, d_r=4),                  # a gs, b gs, d global
        6: dict(a_s=0, a_r=1, b_s=2, b_r=2, d_s=3, d_r=4),                  # a gs, b global, d gs
        7: dict(a_s=0, a_r=0, b_s=1, b_r=2, d_s=3, d_r=4),                  # a global, b gs, d gs
        8: dict(a_s=0, a_r=0, b_s=1, b_r=1, d_s=2, d_r=3),                  # a global, b global, d gs
    }
    return m[int(scenario)]

def group_param_labels(scenario: int, group_name: str):
    scenario = int(scenario)
    idx = scenario_param_indices(scenario)

    def lab(shared, gs, which):
        return shared if gs is False else f"{gs}_{which}"

    # alpha label
    alpha_is_global = (idx["a_s"] == idx["a_r"])
    beta_is_global  = (idx["b_s"] == idx["b_r"])
    delt_is_global  = (idx["d_s"] == idx["d_r"])

    if group_name == "radiation_sensitive":
        a_lab = "a" if alpha_is_global else "a_s"
        b_lab = "b" if beta_is_global else "b_s"
        d_lab = "d" if delt_is_global else "d_s"
    else:
        a_lab = "a" if alpha_is_global else "a_r"
        b_lab = "b" if beta_is_global else "b_r"
        d_lab = "d" if delt_is_global else "d_r"

    return [a_lab, b_lab, d_lab]

def log_likelihood(theta, model_extension, full_data, group, scenario=1):
    ll = 0

    rs = 8.89429860e-02
    rr = 1.15631515e-01
    carrying_capacity = 3.69671552e+03

    variance = theta[-1] ** 2

    idx_map = scenario_param_indices(scenario)
    a_s = theta[idx_map["a_s"]]
    a_r = theta[idx_map["a_r"]]
    b_s = theta[idx_map["b_s"]]
    b_r = theta[idx_map["b_r"]]
    d_s = theta[idx_map["d_s"]]
    d_r = theta[idx_map["d_r"]]

    ndim = 0
    for c_group in group:
        ndim += len(full_data[c_group])

    idx = 0
    for c_group in group:
        for mouse_data in full_data[c_group]:
            treatment_days = mouse_data['treatment_days']
            dose = 2
            time_array = mouse_data['data'][:, 0]

            if c_group == 'radiation_sensitive':
                growth_rate = rs
                a_radiation = a_s
                b_radiation = b_s
                delay = d_s
            elif c_group == 'radiation_resistant':
                growth_rate = rr
                a_radiation = a_r
                b_radiation = b_r
                delay = d_r
            else:
                raise ValueError(f"Unexpected group: {c_group}")

            treatment_params = (a_radiation, b_radiation, delay)
            initial_condition = theta[-(ndim + 1) + idx]

            solution = solve_model(
                model_extension,
                time_array,
                (growth_rate, carrying_capacity, *treatment_params),
                initial_condition,
                treatment_days,
                dose
            )

            observed_volume = mouse_data['data'][:, 1]
            ll += -0.5 * np.sum((solution[:, 0] - observed_volume) ** 2 / variance + np.log(2 * np.pi) + np.log(variance))
            idx += 1

    return ll

def log_prior(theta, l_bound, u_bound):
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0

def log_probability(theta, l_bound, u_bound, model_extension, full_data, group, scenario=1):
    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model_extension, full_data, group, scenario=scenario)

def find_max_time_per_group(full_data):
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

def color_plot_scatter(final_exp, final_model, save, show, group, figure_name, formatter):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig, ax = plt.subplots(figsize=(5, 5))
    c = 1
    all_cccs = []
    for i in range(len(final_exp)):
        all_cccs.append(ccc(final_model[i], final_exp[i]))
        if all_cccs[-1] < 0.8:
            ax.plot(final_exp[i], final_model[i], 'o', color=colors[c%len(colors)])
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
    max_value = max((max(final_exp),max(final_model)))
    max_value = math.ceil(max_value / 1000) * 1000
    ticks = np.linspace(0, max_value, num=int(max_value/1000)+1)  # Adjust num for the desired number of ticks
    
    ax.set_ylim((0, max_value))
    ax.set_xlim((0, max_value))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    
    pccT = pcc(final_model, final_exp)
    cccT = ccc(final_model, final_exp)
    #mapeT = mape(final_model, final_exp)
    all_cccs = np.array(all_cccs)
    if group == "radiation_resistant":
        mapeT = mape(final_model, final_exp)
        ax.text(0.025, .8, 'CCC/PCC = {:.2f}/{:.2f}\nMAPE = {:.2f}%\n CCC = {:.2f}$\pm${:.2f}'.format(cccT, pccT, mapeT, all_cccs.mean(), all_cccs.std()), horizontalalignment='left', transform=ax.transAxes)
    else:
        ax.text(0.025, .8, 'CCC/PCC = {:.2f}/{:.2f}\n CCC = {:.2f}$\pm${:.2f}'.format(cccT, pccT, all_cccs.mean(), all_cccs.std()), horizontalalignment='left', transform=ax.transAxes)
    
    if save:
        plt.savefig(figure_name + '_sp.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close()
    return
        
def plot_maxll_solution(files_location_sensitive, files_location_resistant, full_data, nCols=3, show=True, save=True, fontsize='14', figure_name='maxll_figure'):
    if not show and not save:
        return

    formatter = configure_plot_settings(fontsize)
    plt.rcParams['font.size'] = fontsize
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))

    theta_sensitive = None
    theta_resistant = None

    if files_location_sensitive is not None:
        npzfile_sensitive = np.load(files_location_sensitive)
        theta_sensitive = npzfile_sensitive['pars']

    if files_location_resistant is not None:
        npzfile_resistant = np.load(files_location_resistant)
        theta_resistant = npzfile_resistant['pars']

    max_times = find_max_time_per_group(full_data)

    rs = 8.89429860e-02
    rr = 1.15631515e-01
    carrying_capacity = 3.69671552e+03

    for group in ['radiation_sensitive', 'radiation_resistant']:
        if group == 'radiation_sensitive' and theta_sensitive is not None:
            theta = theta_sensitive
        elif group == 'radiation_resistant' and theta_resistant is not None:
            theta = theta_resistant
        else:
            continue

        nScenarios = len(full_data[group])
        nRows = int(np.ceil(nScenarios / nCols))
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 11, nRows * 5))
        axes = axes.ravel()

        full_exp = []
        full_model = []

        model_extension = "_ExponentialDecayDelay"
        ndim = len(full_data[group])
        treatment_params = tuple(theta[:-(ndim + 1)])

        max_time_group = max_times[group]

        for idx, mouse_data in enumerate(full_data[group]):
            mouse_name = mouse_data['name']
            ax = axes[idx]

            data = mouse_data['data']
            treatment_days = mouse_data['treatment_days']
            dose = 2

            ax.set_ylabel('Tumor Volume (mm³)')
            ax.text(0.05, .9, f'{string.ascii_uppercase[idx]}) {mouse_name}', horizontalalignment='left', transform=ax.transAxes, fontsize=18)
            ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10, label='Tumor Volume')
            ax.set_xlabel('Time (days)')
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim((-1, max_time_group + 1))

            for t_day in treatment_days:
                ax.axvline(x=t_day, color='green', linestyle='--', linewidth=1)

            growth_rate = rs if group == 'radiation_sensitive' else rr
            initial_condition = theta[-(ndim + 1) + idx]

            solution = solve_model(
                model_extension,
                data[:, 0],
                (growth_rate, carrying_capacity, *treatment_params),
                initial_condition,
                treatment_days,
                dose,
                'smooth'
            )

            mask = np.isin(solution[:, 0], data[:, 0])
            matched_solution_times = solution[mask, 0]
            matched_solution_volumes = solution[mask, 1]

            ax.plot(matched_solution_times, matched_solution_volumes, color='black', linestyle='dashed', linewidth=2, label='Model')

            full_model.append(matched_solution_volumes)
            full_exp.append(data[:, 1])

            pccT = pcc(matched_solution_volumes, data[:, 1])
            cccT = ccc(matched_solution_volumes, data[:, 1])

            if group == 'radiation_sensitive':
                ax.text(0.5, .05, f'CCC/PCC = {cccT:.2f}/{pccT:.2f}', horizontalalignment='left', transform=ax.transAxes)
            else:
                mapeT = mape(matched_solution_volumes, data[:, 1])
                ax.text(0.5, .05, 'CCC/PCC/MAPE = {:.2f}/{:.2f}/{:.2f}%'.format(cccT, pccT, mapeT), horizontalalignment='left', transform=ax.transAxes)

        finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, figure_name, formatter)
        color_plot_scatter(full_exp, full_model, save, show, group, figure_name, formatter)

    return

def find_max_initial_condition(full_data):
    maior_ic = -float('inf')
    grupo_maior_ic = None
    camundongo_maior_ic = None
    for group, mice_data in full_data.items():
        for mouse in mice_data:
            ic = mouse['data'][0, 1]
            if ic > maior_ic:
                maior_ic = ic
                grupo_maior_ic = group
                camundongo_maior_ic = mouse['name']
    return maior_ic, grupo_maior_ic, camundongo_maior_ic

max_ic = find_max_initial_condition(full_data)[0]

def define_bounds_labels(model_extension, scenario=1, bool_multiple_mice=False):
    labels = []
    l_bound = []
    u_bound = []

    ic_max = max_ic * 2
    scenario = int(scenario)

    if model_extension == '_ExponentialDecayDelay':
        idx = scenario_param_indices(scenario)

        alpha_global = (idx["a_s"] == idx["a_r"])
        beta_global  = (idx["b_s"] == idx["b_r"])
        delt_global  = (idx["d_s"] == idx["d_r"])

        if alpha_global:
            labels += ["a"]
            l_bound += [0.0]
            u_bound += [0.1]
        else:
            labels += ["a_s", "a_r"]
            l_bound += [0.0, 0.0]
            u_bound += [0.1, 0.1]

        if beta_global:
            labels += ["b"]
            l_bound += [0.0]
            u_bound += [2.0]
        else:
            labels += ["b_s", "b_r"]
            l_bound += [0.0, 0.0]
            u_bound += [0.1, 2.0]

        if delt_global:
            labels += ["d"]
            l_bound += [0.0]
            u_bound += [10.0]
        else:
            labels += ["d_s", "d_r"]
            l_bound += [0.0, 0.0]
            u_bound += [10.0, 10.0]

    if bool_multiple_mice:
        for i in range(group_size):
            labels.append("ic" + str(i + 1))
            l_bound.append(0.0)
            u_bound.append(ic_max)
        labels.append("std")
        l_bound.append(0.1)
        u_bound.append(1000.0)
    else:
        labels.append("ic")
        l_bound.append(0.0)
        u_bound.append(ic_max)
        labels.append("std")
        l_bound.append(0.1)
        u_bound.append(1000.0)

    return l_bound, u_bound, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce1", action="store_true")
    parser.add_argument("--sce2", action="store_true")
    parser.add_argument("--sce3", action="store_true")
    parser.add_argument("--sce4", action="store_true")
    parser.add_argument("--sce5", action="store_true")
    parser.add_argument("--sce6", action="store_true")
    parser.add_argument("--sce7", action="store_true")
    parser.add_argument("--sce8", action="store_true")
    args = parser.parse_args()

    selected_scenarios = []
    if args.sce1: selected_scenarios.append(1)
    if args.sce2: selected_scenarios.append(2)
    if args.sce3: selected_scenarios.append(3)
    if args.sce4: selected_scenarios.append(4)
    if args.sce5: selected_scenarios.append(5)
    if args.sce6: selected_scenarios.append(6)
    if args.sce7: selected_scenarios.append(7)
    if args.sce8: selected_scenarios.append(8)
    if not selected_scenarios:
        selected_scenarios = [1]

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

    run_calibration = True
    if run_calibration:
        for model_extension in ['_ExponentialDecayDelay']:
            original_group = ['radiation_sensitive', 'radiation_resistant']

            group_size = 0
            new_group = []
            for group in original_group:
                group_size += len(full_data[group])
                new_group.append(group)

            group = new_group
            num_ic_sensitive = len(full_data['radiation_sensitive'])
            num_ic_resistant = len(full_data['radiation_resistant'])

            for scenario in selected_scenarios:
                version = scenario_tag(scenario)
                base_n = scenario_base_dim(scenario)
                idx_map = scenario_param_indices(scenario)

                std_idx = base_n + num_ic_sensitive + num_ic_resistant
                ics_s_idx = list(range(base_n, base_n + num_ic_sensitive))
                ics_r_idx = list(range(base_n + num_ic_sensitive, base_n + num_ic_sensitive + num_ic_resistant))

                l_bound, u_bound, labels = define_bounds_labels(model_extension, scenario=scenario, bool_multiple_mice=True)
                ndim = len(l_bound)
                nwalkers = 2 * ndim

                filename = f"./reinitiate_files/calibration_{model_extension}_{version}.h5"

                if not os.path.exists(filename):
                    print("HDF5 file not found. Creating a new file....")
                    backend = emcee.backends.HDFBackend(filename)
                    backend.reset(nwalkers, ndim)
                    pos = np.zeros((nwalkers, ndim))
                    for i in range(ndim):
                        pos[:, i] = np.random.uniform(low=l_bound[i], high=u_bound[i], size=nwalkers)
                else:
                    print(f"HDF5 file found: {filename}")
                    backend = emcee.backends.HDFBackend(filename)
                    print(f"Current progress in backend: {backend.iteration} iterations.")
                    pos = None

                additional_chain_size = 10
                print(f'Calibrating {group}, using model {model_extension.split("_")[-1]} | {version}')

                with Pool(processes=cpu_count()) as pool:
                    sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability,
                        args=(l_bound, u_bound, model_extension, full_data, group, scenario),
                        pool=pool,
                        backend=backend
                    )
                    if backend.iteration > 0:
                        sampler.run_mcmc(None, additional_chain_size, progress=True)
                    else:
                        sampler.run_mcmc(pos, additional_chain_size, progress=True)

                flat_ll = sampler.get_log_prob(flat=True)
                flat_chain = sampler.get_chain(flat=True)
                max_pos = flat_ll.argmax(axis=0)
                best_pars = flat_chain[max_pos]

                total_size = ((additional_chain_size - additional_chain_size // 4) * ndim * 2)
                goal_size = 10000
                scale_corner = max(1, total_size // goal_size)
                save_chain = sampler.get_chain(discard=additional_chain_size // 4, flat=True, thin=scale_corner)

                print(f"Total steps completed: {backend.iteration}")

                for g in original_group:
                    if g == 'radiation_sensitive':
                        cols = [idx_map["a_s"], idx_map["b_s"], idx_map["d_s"]] + ics_s_idx + [std_idx]
                        labels_sensitive = group_param_labels(scenario, g) + [f"ic{i+1}" for i in range(num_ic_sensitive)] + ["std"]

                        chain_sensitive = save_chain[:, cols]
                        best_pars_sensitive = best_pars[cols]

                        np.savetxt(f'./Output_Calibration/multi_chain_radiation_sensitive{model_extension}_{version}.gz', chain_sensitive)
                        np.savez(f'./Output_Calibration/multi_ll_pars_radiation_sensitive{model_extension}_{version}.npz', max_ll=max(flat_ll), pars=best_pars_sensitive)

                        plot_maxll_solution(
                            f'./Output_Calibration/multi_ll_pars_radiation_sensitive{model_extension}_{version}.npz',
                            None,
                            full_data,
                            nCols=4,
                            show=False,
                            save=True,
                            figure_name=f'./Output_Calibration/multi_max_ll_radiation_sensitive{model_extension}_{version}'
                        )

                        fig_sensitive = corner.corner(chain_sensitive, labels=labels_sensitive, truths=best_pars_sensitive)
                        plt.savefig(f'./Output_Calibration/multi_corner_radiation_sensitive{model_extension}_{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                        plt.close()

                        fig, axes = plt.subplots(nrows=len(labels_sensitive), ncols=1, figsize=(10, len(labels_sensitive) * 2), sharex=True)
                        axes = axes.ravel()
                        samples = sampler.get_chain()
                        samples_sensitive = samples[:, :, cols]
                        for i in range(len(labels_sensitive)):
                            ax = axes[i]
                            ax.plot(samples_sensitive[:, :, i], "k", alpha=0.3)
                            ax.set_xlim(0, len(samples_sensitive))
                            ax.set_ylabel(labels_sensitive[i])
                            ax.yaxis.set_label_coords(-0.1, 0.5)
                        axes[-1].set_xlabel("step number")
                        plt.savefig(f'./Output_Calibration/multi_chain_radiation_sensitive{model_extension}_{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                        plt.close()

                    elif g == 'radiation_resistant':
                        cols = [idx_map["a_r"], idx_map["b_r"], idx_map["d_r"]] + ics_r_idx + [std_idx]
                        labels_resistant = group_param_labels(scenario, g) + [f"ic{i+1}" for i in range(num_ic_resistant)] + ["std"]

                        chain_resistant = save_chain[:, cols]
                        best_pars_resistant = best_pars[cols]

                        np.savetxt(f'./Output_Calibration/multi_chain_radiation_resistant{model_extension}_{version}.gz', chain_resistant)
                        np.savez(f'./Output_Calibration/multi_ll_pars_radiation_resistant{model_extension}_{version}.npz', max_ll=max(flat_ll), pars=best_pars_resistant)

                        plot_maxll_solution(
                            None,
                            f'./Output_Calibration/multi_ll_pars_radiation_resistant{model_extension}_{version}.npz',
                            full_data,
                            nCols=4,
                            show=False,
                            save=True,
                            figure_name=f'./Output_Calibration/multi_max_ll_radiation_resistant{model_extension}_{version}'
                        )

                        fig_resistant = corner.corner(chain_resistant, labels=labels_resistant, truths=best_pars_resistant)
                        plt.savefig(f'./Output_Calibration/multi_corner_radiation_resistant{model_extension}_{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                        plt.close()

                        fig, axes = plt.subplots(nrows=len(labels_resistant), ncols=1, figsize=(10, len(labels_resistant) * 2), sharex=True)
                        axes = axes.ravel()
                        samples = sampler.get_chain()
                        samples_resistant = samples[:, :, cols]
                        for i in range(len(labels_resistant)):
                            ax = axes[i]
                            ax.plot(samples_resistant[:, :, i], "k", alpha=0.3)
                            ax.set_xlim(0, len(samples_resistant))
                            ax.set_ylabel(labels_resistant[i])
                            ax.yaxis.set_label_coords(-0.1, 0.5)
                        plt.savefig(f'./Output_Calibration/multi_chain_radiation_resistant{model_extension}_{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                        plt.close()
