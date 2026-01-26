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
import pdse_project as pdse
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


@jit(nopython=True)
def model_log(y, t, growth_rate, carrying_capacity):
    tumor = y
    tumorVolume = growth_rate * tumor * (1 - tumor/carrying_capacity)
    return tumorVolume

def solve_model(model_extension, time_array, parameters, initial_condition, type_sol='data'):
    model_name = 'model' + model_extension
    model_func = globals()[model_name]
    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p-bgn_p)/0.1) + 1)
        sol = odeint(model_func, t=time_array, y0=[initial_condition], args=parameters, mxstep=2000)
        return np.column_stack((time_array, sol))
    else:
        return odeint(model_func, t=time_array, y0=[initial_condition], args=parameters, mxstep=2000)

def scenario_tag(scenario: int) -> str:
    return f"V{int(scenario)}"

def scenario_base_dim(scenario: int) -> int:
    return {1: 4, 2: 3, 3: 3, 4: 2}[int(scenario)]

def scenario_param_indices(scenario: int):
    mapping = {
        1: dict(r_s=0, r_r=1, k_s=2, k_r=3),
        2: dict(r_s=0, r_r=1, k_s=2, k_r=2),
        3: dict(r_s=0, r_r=0, k_s=1, k_r=2),
        4: dict(r_s=0, r_r=0, k_s=1, k_r=1),
    }
    return mapping[int(scenario)]

def log_likelihood(theta, model_extension, full_data, group, scenario=1):
    ll = 0
    variance = theta[-1] ** 2

    if int(scenario) == 1:
        rs, rr, ks, kr = theta[0], theta[1], theta[2], theta[3]
    elif int(scenario) == 2:
        rs, rr, kglob = theta[0], theta[1], theta[2]
        ks, kr = kglob, kglob
    elif int(scenario) == 3:
        rglob, ks, kr = theta[0], theta[1], theta[2]
        rs, rr = rglob, rglob
    elif int(scenario) == 4:
        rglob, kglob = theta[0], theta[1]
        rs, rr = rglob, rglob
        ks, kr = kglob, kglob
    else:
        raise ValueError("scenario must be one of {1,2,3,4}")

    ndim = 0
    for c_group in group:
        ndim += len(full_data[c_group])

    idx = 0
    for c_group in group:
        for mouse_data in full_data[c_group]:
            time_array = mouse_data['data'][:, 0]

            if c_group == 'control_sensitive':
                growth_rate = rs
                carrying_capacity = ks
            elif c_group == 'control_resistant':
                growth_rate = rr
                carrying_capacity = kr
            else:
                raise ValueError(f"Unexpected group: {c_group}")

            parameters = (growth_rate, carrying_capacity)

            initial_condition = theta[-(ndim+1) + idx]
            solution = solve_model(model_extension, time_array, parameters, initial_condition)

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

def plot_maxll_solution(files_location_sensitive, files_location_resistant, full_data, nCols=3, show=True, save=True, fontsize='14', figure_name='maxll_figure'):
    if not show and not save:
        return

    formatter = pdse.configure_plot_settings(fontsize)
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

    for group in ['control_sensitive', 'control_resistant']:
        if group == 'control_sensitive' and theta_sensitive is not None:
            theta = theta_sensitive
        elif group == 'control_resistant' and theta_resistant is not None:
            theta = theta_resistant
        else:
            continue

        nScenarios = len(full_data[group])
        nRows = int(np.ceil(nScenarios / nCols))
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 11, nRows * 5))
        axes = axes.ravel()

        full_exp = []
        full_model = []

        model_extension = "_log"
        ndim = len(full_data[group])
        max_time_group = max_times[group]

        for idx, mouse_data in enumerate(full_data[group]):
            mouse_name = mouse_data['name']
            ax = axes[idx]
            data = mouse_data['data']

            ax.set_ylabel('Tumor Volume (mm³)')
            ax.text(0.05, .9, f'{string.ascii_uppercase[idx]}) {mouse_name}', horizontalalignment='left', transform=ax.transAxes, fontsize=18)
            ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10, label='Tumor Volume')
            ax.set_xlabel('Time (days)')
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim((-1, max_time_group + 1))

            parameters = tuple(theta[:-(ndim+1)])
            initial_condition = theta[-(ndim+1) + idx]

            solution = solve_model(model_extension, data[:, 0], parameters, initial_condition, "smooth")

            mask = np.isin(solution[:, 0], data[:, 0])
            matched_solution_times = solution[mask, 0]
            matched_solution_volumes = solution[mask, 1]

            ax.plot(matched_solution_times, matched_solution_volumes, color='black', linestyle='dashed', linewidth=2, label='Model')

            full_model.append(matched_solution_volumes)
            full_exp.append(data[:, 1])

            pccT = pdse.pcc(matched_solution_volumes, data[:, 1])
            cccT = pdse.ccc(matched_solution_volumes, data[:, 1])
            mapeT = pdse.mape(matched_solution_volumes, data[:, 1])

            ax.text(0.5, .05, f'CCC/PCC/MAPE = {cccT:.2f}/{pccT:.2f}/{mapeT:.2f}%', horizontalalignment='left', transform=ax.transAxes)

        pdse.finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, figure_name, formatter)
        pdse.color_plot_scatter(full_exp, full_model, save, show, figure_name, formatter)

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

    if model_extension == '_log':
        scenario = int(scenario)
        if scenario == 1:
            labels = ["r_s", "r_r", "K_s", "K_r"]
            l_bound = [0.0, 0.0, 100.0, 100.0]
            u_bound = [1.0, 1.0, 20000.0, 20000.0]
        elif scenario == 2:
            labels = ["r_s", "r_r", "K"]
            l_bound = [0.0, 0.0, 100.0]
            u_bound = [1.0, 1.0, 20000.0]
        elif scenario == 3:
            labels = ["r", "K_s", "K_r"]
            l_bound = [0.0, 100.0, 100.0]
            u_bound = [1.0, 20000.0, 20000.0]
        elif scenario == 4:
            labels = ["r", "K"]
            l_bound = [0.0, 100.0]
            u_bound = [1.0, 20000.0]
        else:
            raise ValueError("scenario must be one of {1,2,3,4}")

    if bool_multiple_mice:
        for i in range(group_size):
            labels.append("ic" + str(i+1))
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
    parser.add_argument("--sce1", action="store_true", help="Run Scenario 1: r group-specific, K group-specific")
    parser.add_argument("--sce2", action="store_true", help="Run Scenario 2: r group-specific, K global")
    parser.add_argument("--sce3", action="store_true", help="Run Scenario 3: r global, K group-specific")
    parser.add_argument("--sce4", action="store_true", help="Run Scenario 4: r global, K global")
    args = parser.parse_args()

    selected_scenarios = [2]
    if args.sce1: selected_scenarios.append(1)
    if args.sce2: selected_scenarios.append(2)
    if args.sce3: selected_scenarios.append(3)
    if args.sce4: selected_scenarios.append(4)
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
        for model_extension in ['_log']:
            original_group = ['control_sensitive', 'control_resistant']

            group_size = 0
            new_group = []
            for group in ['control_sensitive', 'control_resistant']:
                group_size += len(full_data[group])
                new_group.append(group)

            group = new_group
            num_ic_sensitive = len(full_data['control_sensitive'])
            num_ic_resistant = len(full_data['control_resistant'])

            for scenario in selected_scenarios:
                tag = scenario_tag(scenario)
                base_n = scenario_base_dim(scenario)
                idx_map = scenario_param_indices(scenario)
                std_idx = base_n + num_ic_sensitive + num_ic_resistant
                ics_s_idx = list(range(base_n, base_n + num_ic_sensitive))
                ics_r_idx = list(range(base_n + num_ic_sensitive, base_n + num_ic_sensitive + num_ic_resistant))

                l_bound, u_bound, labels = define_bounds_labels(model_extension, scenario=scenario, bool_multiple_mice=True)
                ndim = len(l_bound)
                nwalkers = 2 * ndim

                filename = f"global_calibration_{tag}{model_extension}.h5"

                if not os.path.exists(filename):
                    print("HDF5 file not found. Creating a new file...")
                    backend = emcee.backends.HDFBackend(filename)
                    backend.reset(nwalkers, ndim)
                    pos = np.zeros((nwalkers, ndim))
                    for i in range(ndim):
                        pos[:, i] = np.random.uniform(low=l_bound[i], high=u_bound[i], size=nwalkers)
                else:
                    print(f"HDF5 file found: {filename}")
                    backend = emcee.backends.HDFBackend(filename)
                    print(f"Current progress on the backend: {backend.iteration} iterations.")
                    if backend.iteration == 0:
                        backend.reset(nwalkers, ndim)
                        pos = np.zeros((nwalkers, ndim))
                        for i in range(ndim):
                            pos[:, i] = np.random.uniform(low=l_bound[i], high=u_bound[i], size=nwalkers)
                    else:
                        pos = None

                additional_chain_size = 100
                print(f'Calibrating {group}, using model {model_extension.split("_")[-1]} | {tag}')

                with Pool() as pool:
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
                    if g == 'control_sensitive':
                        r_idx = idx_map["r_s"]
                        k_idx = idx_map["k_s"]
                        cols = [r_idx, k_idx] + ics_s_idx + [std_idx]

                        labels_sensitive = ["r", "K"] + [f"ic{i+1}" for i in range(num_ic_sensitive)] + ["std"]
                        chain_sensitive = save_chain[:, cols]
                        best_pars_sensitive = best_pars[cols]

                        np.savetxt(f'./Output_Calibration/multi_chain_control_sensitive{model_extension}_{tag}.gz', chain_sensitive)
                        np.savez(f'./Output_Calibration/multi_ll_pars_control_sensitive{model_extension}_{tag}.npz', max_ll=max(flat_ll), pars=best_pars_sensitive)

                        plot_maxll_solution(
                            f'./Output_Calibration/multi_ll_pars_control_sensitive{model_extension}_{tag}.npz',
                            None,
                            full_data,
                            nCols=4,
                            show=False,
                            save=True,
                            figure_name=f'./Output_Calibration/multi_max_ll_control_sensitive{model_extension}_{tag}'
                        )

                        fig_sensitive = corner.corner(chain_sensitive, labels=labels_sensitive)
                        plt.savefig(f'./Output_Calibration/multi_corner_control_sensitive{model_extension}_{tag}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
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
                        plt.savefig(f'./Output_Calibration/multi_chain_control_sensitive{model_extension}_{tag}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                        plt.close()

                    elif g == 'control_resistant':
                        r_idx = idx_map["r_r"]
                        k_idx = idx_map["k_r"]
                        cols = [r_idx, k_idx] + ics_r_idx + [std_idx]

                        labels_resistant = ["r", "K"] + [f"ic{i+1}" for i in range(num_ic_resistant)] + ["std"]
                        chain_resistant = save_chain[:, cols]
                        best_pars_resistant = best_pars[cols]

                        np.savetxt(f'./Output_Calibration/multi_chain_control_resistant{model_extension}_{tag}.gz', chain_resistant)
                        np.savez(f'./Output_Calibration/multi_ll_pars_control_resistant{model_extension}_{tag}.npz', max_ll=max(flat_ll), pars=best_pars_resistant)

                        plot_maxll_solution(
                            None,
                            f'./Output_Calibration/multi_ll_pars_control_resistant{model_extension}_{tag}.npz',
                            full_data,
                            nCols=4,
                            show=False,
                            save=True,
                            figure_name=f'./Output_Calibration/multi_max_ll_control_resistant{model_extension}_{tag}'
                        )

                        fig_resistant = corner.corner(chain_resistant, labels=labels_resistant)
                        plt.savefig(f'./Output_Calibration/multi_corner_control_resistant{model_extension}_{tag}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
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
                        plt.savefig(f'./Output_Calibration/multi_chain_control_resistant{model_extension}_{tag}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                        plt.close()
