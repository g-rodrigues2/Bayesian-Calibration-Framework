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
import matplotlib.colors as mcolors
from scipy.integrate import odeint
import seaborn as sns
import scipy.stats as stats
import sys

# Find and sort all text files in the ./data directory
data_files = sorted(glob.glob("./data/*.txt"))

# Dictionary to store the parsed data, grouped by data type
full_data = {}

# Loop through each file in the data_files list
for file in data_files:
    # Extract the data type (treatment and cell type) from the file name
    data_type = os.path.basename(file).split('/')[-1].split('_c')[0]
    
    # Extract the mouse name (cohort identifier) from the file name
    mouse_name = 'c' + file.split('/')[-1].split('_c')[-1].split('_t')[0]
    
    # Extract the treatment days from the file name (as integers)
    t_days = np.array([int(t) for t in file.split('/')[-1].split('_c')[-1].split('.txt')[0].split('_SUV')[0].split('_t')[1:]])
 
    # Por fim, o valor de SUV está logo após '_SUV'
    suv_value = float(os.path.splitext(os.path.basename(file))[0].split("_SUV")[-1])

    # Load the measurement data from the file (time, tumor volume)
    data = np.loadtxt(file)
    
    # Extract the specific treatment from the data type (e.g., radiation or control)
    treatment = os.path.basename(data_type).split('_')[0]
    
    # If this data type hasn't been seen before, create a new list in full_data
    if data_type not in full_data:
        full_data[data_type] = []
    
    # Append the mouse's data (including name, tumor measurements, treatment, and treatment days) to the appropriate group
    full_data[data_type].append({
        'name': mouse_name,
        'data': data,
        'treatment': treatment,
        'treatment_days': t_days,
        'SUV': suv_value
    })

    # Concordance Correlation Coefficient
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
def model_ExponentialDecay(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    # Crescimento logístico
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    # Efeito da imunoterapia
    treatment_effect = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            treatment_effect += a_imuno * dose * np.exp(-b_imuno * (t - tau))

    # Ajuste do volume tumoral pelo tratamento
    tumor_volume -= treatment_effect * tumor

    return tumor_volume


@jit(nopython=True)
def model_CumulativeDose(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    cumulative_dose = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            cumulative_dose += dose
            treatment_effect += a_imuno * cumulative_dose * np.exp(-b_imuno * (t - tau))

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_AccelED(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            treatment_effect += a_imuno * dose * np.exp(-b_imuno * (t - tau)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_AccelCD(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    cumulative_dose = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            cumulative_dose += dose
            treatment_effect += a_imuno * cumulative_dose * np.exp(-b_imuno * (t - tau)**2)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume


@jit(nopython=True)
def model_MichaelisMenten(y, t, growth_rate, carrying_capacity, a_imuno, b_imuno, tau_imuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    for dose, tau in zip(dose_schedule, tau_imuno):
        if t >= tau:
            treatment_effect += (a_imuno * dose) / (1 + b_imuno * dose)


    tumor_volume -= treatment_effect * tumor  

    return tumor_volume


@jit(nopython=True)
def model_dePillisRadunskayaLaw(y, t, growth_rate, carrying_capacity, d, s, lam, tau_immuno, dose_schedule):

    tumor = max(y[0], 1e-12)
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0.0
    for i in range(len(dose_schedule)):
        dose = dose_schedule[i]
        tau = tau_immuno[i]
        if t >= tau:
            effect = d * (dose ** lam) / (s * (tumor ** lam) + dose ** lam)
            treatment_effect += effect

    tumor_volume -= treatment_effect * tumor

    return tumor_volume

@jit(nopython=True)
def model_CDwithLogFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, steepness, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            logistic_factor = 1 / (1 + np.exp(-steepness * time_since_treatment))
            cumulative_dose += dose
            treatment_effect += a_immuno * cumulative_dose * logistic_factor * np.exp(-b_immuno * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_CDwithExpFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, rate, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            cumulative_dose += dose
            treatment_effect += a_immuno * cumulative_dose * exponential_factor * np.exp(-b_immuno * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_CDwithSinFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, frequency, tau_immuno, dose_schedule):
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    cumulative_dose = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            cumulative_dose += dose
            treatment_effect += a_immuno * cumulative_dose * sinusoidal_factor * np.exp(-b_immuno * time_since_treatment)

    tumor_volume -= treatment_effect * tumor
    return tumor_volume

@jit(nopython=True)
def model_EDwithLogFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, steepness, tau_immuno, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau: 
            time_since_treatment = t - tau
            logistic_factor = 1 / (1 + np.exp(-steepness * time_since_treatment))
            treatment_effect += a_immuno * dose * logistic_factor * np.exp(-b_immuno * time_since_treatment)
    tumor_volume -= treatment_effect * tumor 
    return tumor_volume

@jit(nopython=True)
def model_EDwithExpFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, rate, tau_immuno, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau:
            time_since_treatment = t - tau
            exponential_factor = 1 - np.exp(-rate * time_since_treatment)
            treatment_effect += a_immuno * dose *  exponential_factor * np.exp(-b_immuno * time_since_treatment)

    tumor_volume -= treatment_effect * tumor 
    return tumor_volume

@jit(nopython=True)
def model_EDwithSinFactor(y, t, growth_rate, carrying_capacity, a_immuno, b_immuno, frequency, tau_immuno, dose_schedule):

    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_immuno):
        if t >= tau: 
            time_since_treatment = t - tau
            sinusoidal_factor = 0.5 * (1 + np.sin(frequency * time_since_treatment))
            treatment_effect += a_immuno * dose *  sinusoidal_factor * np.exp(-b_immuno * time_since_treatment)
    tumor_volume -= treatment_effect * tumor 
    return tumor_volume

def solve_model(model_extension, time_array, parameters, initial_condition, treatment_days, dose, type_sol='data'):
    model_name = 'model' + model_extension
    model_func = globals()[model_name]

    tau_imuno = treatment_days
    dose_schedule = [dose] * len(tau_imuno) 

    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p - bgn_p) / 0.1) + 1)
        sol = odeint(model_func, t=time_array, y0=[initial_condition], args=(*parameters, tau_imuno, dose_schedule), mxstep=10000)
        return np.column_stack((time_array, sol))
    else:
        return odeint(model_func, t=time_array, y0=[initial_condition], args=(*parameters, tau_imuno, dose_schedule), mxstep=10000)


def log_likelihood(theta, model_extension, full_data, group):
    ll = 0
    growth_rate = 8.89429860e-02 
    carrying_capacity = 3.69671552e+03  

    variance = theta[-1] ** 2

    ndim = len(full_data[group])
    for idx, mouse_data in enumerate(full_data[group]):        
        
        treatment_days = mouse_data['treatment_days'] 
        dose = 1.0 
        time_array = mouse_data['data'][:, 0]

        treatment_params = theta[:-(ndim + 1)] 
        initial_condition = theta[-(ndim + 1) + idx] 

        solution = solve_model(model_extension, time_array, (growth_rate, carrying_capacity, *treatment_params), 
                               initial_condition, treatment_days, dose)

        observed_volume = mouse_data['data'][:, 1] 

        ll += -0.5 * np.sum((solution[:, 0] - observed_volume) ** 2 / variance + np.log(2 * np.pi) + np.log(variance))

    return ll

def log_prior(theta, l_bound, u_bound):

    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0

def log_probability(theta, l_bound, u_bound, model_extension, full_data, group):

    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf  

    return lp + log_likelihood(theta, model_extension, full_data, group)

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

def color_plot_scatter(final_exp, final_model, save, show, figure_name, formatter):
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
    mapeT = mape(final_model, final_exp)
    all_cccs = np.array(all_cccs)
    ax.text(0.025, .8, 'CCC/PCC = {:.2f}/{:.2f}\nMAPE = {:.2f}%\n CCC = {:.2f}$\pm${:.2f}'.format(cccT, pccT, mapeT, all_cccs.mean(), all_cccs.std()), horizontalalignment='left', transform=ax.transAxes)
    # ax.text(0.025, .8, 'CCC/PCC = {:.2f}/{:.2f}\n CCC = {:.2f}$\pm${:.2f}'.format(cccT, pccT, all_cccs.mean(), all_cccs.std()), horizontalalignment='left', transform=ax.transAxes)
    
    if save:
        plt.savefig(figure_name + '_sp.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close()
    return

def plot_maxll_solution_global(files_location, full_data, nCols=3, show=True, save=True, fontsize='14', figure_name='maxll_figure'):
    if not show and not save:
        return 
    
    all_chains = sorted(glob.glob(files_location))
    formatter = configure_plot_settings(fontsize)
    plt.rcParams['font.size'] = fontsize
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))

    best_chain_per_group = {}
    max_ll_per_group = {}

    for chain in all_chains:
        npzfile = np.load(chain)
        max_ll = npzfile['max_ll']

        chain_split = chain.split('_')
        group = f"{chain_split[4]}_{chain_split[5]}" 

        if group not in max_ll_per_group or max_ll > max_ll_per_group[group]:
            max_ll_per_group[group] = max_ll
            best_chain_per_group[group] = chain

    max_times = find_max_time_per_group(full_data)

    growth_rate = 8.89429860e-02
    carrying_capacity = 3.69671552e+03

    for group, best_chain in best_chain_per_group.items():
        print(f"Processing: Group = {group}, Best fit found with max_ll = {max_ll_per_group[group]}")

        npzfile = np.load(best_chain)
        theta = npzfile['pars']
        print(theta)

        nScenarios = len(full_data[group])
        nRows = int(np.ceil(nScenarios / nCols))
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 11, nRows * 5))
        axes = axes.ravel()

        full_exp = []
        full_model = []

        model_extension = f"_{best_chain.split('_')[-1].split('.')[0]}" 

        ndim = len(full_data[group])
        scale = 1
        print(f'Size pars = {len(theta)}, animals = {ndim}')
        max_time_group = max_times[group]

        for idx, mouse_data in enumerate(full_data[group]):
            mouse_name = mouse_data['name']
            ax = axes[idx]

            data = mouse_data['data']
            treatment_days = mouse_data['treatment_days']
 
            dose = 1.0 
            ax.set_ylabel('Tumor Volume (mm³)')
            ax.text(0.05, .9, f'{string.ascii_uppercase[idx]}) {mouse_name}', horizontalalignment='left', transform=ax.transAxes, fontsize=18)

            ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10, label='Tumor Volume')
            ax.set_xlabel('Time (days)')
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim((-1, max_time_group + 1))
            for t_day in treatment_days:
                ax.axvline(x=t_day, color='red', linestyle='--', linewidth=1)

            treatment_params = tuple(theta[:-(ndim + 1)])
            print(f"Calibrated treatment parameters: {treatment_params}")
            initial_condition = theta[-(ndim + 1) + idx]

            solution = solve_model(
                model_extension, 
                data[:, 0], 
                (growth_rate, carrying_capacity, *treatment_params),  # Adicionar os parâmetros fixos
                initial_condition, 
                treatment_days, 
                dose,
                type_sol= "smooth"
            )

            mask = np.isin(solution[:, 0], data[:, 0])
            matched_solution_times = solution[mask, 0]
            matched_solution_volumes = solution[mask, 1]

            ax.plot(solution[:,0], solution[:,1], color='black', linestyle='dashed', linewidth=2, label='Model')
        

            full_model.append(matched_solution_volumes)
            full_exp.append(data[:, 1])

            pccT = pcc(matched_solution_volumes, data[:, 1])
            cccT = ccc(matched_solution_volumes, data[:, 1])
            mapeT = mape(matched_solution_volumes, data[:, 1])

            ax.text(0.5, .05, 'CCC/PCC/MAPE = {:.2f}/{:.2f}/{:.2f}%'.format(cccT, pccT, mapeT), horizontalalignment='left', transform=ax.transAxes)

            if idx == 0:
                ax.legend(loc=1)

        finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, figure_name, formatter)
        color_plot_scatter(full_exp, full_model, save, show, figure_name, formatter)

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

max_ic = find_max_initial_condition(full_data)
max_ic = max_ic[0]

def define_bounds_labels(model_extension, bool_multiple_mice = False):
    labels = []
    l_bound = []
    u_bound = []
    ic_max = max_ic*2
    if model_extension == '_ExponentialDecay':
        labels = ["alpha_immuno", "beta_immuno"]
        l_bound = [0.01, 1.5]   
        u_bound = [1.0, 3.0]   

    elif model_extension == '_CumulativeDose':
        labels = ["alpha_immuno", "beta_immuno"]
        l_bound = [0.01, 5.0]   
        u_bound = [1.0, 8.0]  

    elif model_extension == '_AccelED':
        labels = ["alpha_immuno", "beta_immuno"]
        l_bound = [0.01, 1.5]   
        u_bound = [1.0, 5.0] 

    elif model_extension == '_AccelCD':
        labels = ["alpha_immuno", "beta_immuno"]
        l_bound = [0.01, 1.5]   
        u_bound = [1.0, 5.0] 

    elif model_extension == '_MichaelisMenten':
        labels = ["alpha_immuno", "beta_immuno"]
        l_bound = [0.0, 20.0]
        u_bound = [1.0, 60.0]

    elif model_extension == '_dePillisRadunskayaLaw':
        labels = ["d", "s", "lam"]
        l_bound = [0.01, 0.01, 0.01]
        u_bound = [2.0, 2.0, 2.0]

    elif model_extension == '_CDwithLogFactor':
        labels = ["alpha_immuno", "beta_immuno", "steepness"]
        l_bound = [0.0, 2.0, 0.0]
        u_bound = [1.0, 6.5, 1.0] 

    elif model_extension == '_CDwithExpFactor':
        labels = ["alpha_immuno", "beta_immuno", "rate"]
        l_bound = [0.0, 0.0, 0.0]
        u_bound = [1.0, 2.5, 1.0] 

    elif model_extension == '_CDwithSinFactor':
        labels = ["alpha_immuno", "beta_immuno", "frequency"]
        l_bound = [0.0, 0.0, 0.0]
        u_bound = [1.0, 7.0, 1.0]  

    elif model_extension == '_EDwithLogFactor':
        labels = ["alpha_immuno", "beta_immuno", "steepness"]
        l_bound = [0.0, 0.0, 0.0]
        u_bound = [1.0, 2.0, 1.0] 

    elif model_extension == '_EDwithExpFactor':
        labels = ["alpha_immuno", "beta_immuno", "rate"]
        l_bound = [0.0, 0.0, 0.0]
        u_bound = [1.0, 1.0, 1.0] 

    elif model_extension == '_EDwithSinFactor':
        labels = ["alpha_immuno", "beta_immuno", "frequency"]
        l_bound = [0.0, 0.0, 0.0]
        u_bound = [1.0, 3.0, 1.0]     

    if bool_multiple_mice:       
        for i in range(group_size):
            labels.append("ic" + str(i+1))
            l_bound.append(60.0)
            u_bound.append(ic_max)  # +30 para permitir variação
        labels.append("std")
        l_bound.append(0.0)
        u_bound.append(200.0)
    else:
        labels.append("ic")
        l_bound.append(80.0)
        u_bound.append(ic_max)
        labels.append("std")
        l_bound.append(0.0)
        u_bound.append(200.0)
    return l_bound, u_bound, labels


# # Main function
if __name__ == "__main__":
    # Find and sort all text files in the ./data directory
    data_files = sorted(glob.glob("./data/*.txt"))

    # Dictionary to store the parsed data, grouped by data type
    full_data = {}

    # Loop through each file in the data_files list
    for file in data_files:
    # Extract the data type (treatment and cell type) from the file name
        data_type = os.path.basename(file).split('/')[-1].split('_c')[0]
        
        # Extract the mouse name (cohort identifier) from the file name
        mouse_name = 'c' + file.split('/')[-1].split('_c')[-1].split('_t')[0]
        
        # Extract the treatment days from the file name (as integers)
        t_days = np.array([int(t) for t in file.split('/')[-1].split('_c')[-1].split('.txt')[0].split('_SUV')[0].split('_t')[1:]])
    
        # Por fim, o valor de SUV está logo após '_SUV'
        suv_value = float(os.path.splitext(os.path.basename(file))[0].split("_SUV")[-1])

        # Load the measurement data from the file (time, tumor volume)
        data = np.loadtxt(file)
        
        # Extract the specific treatment from the data type (e.g., radiation or control)
        treatment = os.path.basename(data_type).split('_')[0]
        
        # If this data type hasn't been seen before, create a new list in full_data
        if data_type not in full_data:
            full_data[data_type] = []
        
        # Append the mouse's data (including name, tumor measurements, treatment, and treatment days) to the appropriate group
        full_data[data_type].append({
            'name': mouse_name,
            'data': data,
            'treatment': treatment,
            'treatment_days': t_days,
            'SUV': suv_value
        })

    run_calibration = True
    if run_calibration:
        for model_extension in [
                                '_ExponentialDecay',
                                '_CumulativeDose',
                                '_AccelED',
                                '_AccelCD',
                                '_MichaelisMenten',
                                '_dePillisRadunskayaLaw',
                                '_CDwithLogFactor',
                                '_CDwithExpFactor',
                                '_CDwithSinFactor',
                                '_EDwithLogFactor',
                                '_EDwithExpFactor',
                                '_EDwithSinFactor',       
                            ]:
            

            for group in ['io_sensitive']:
                print(f"Solving group {group}")

                group_size = len(full_data[group])
                
                l_bound, u_bound, labels = define_bounds_labels(model_extension, bool_multiple_mice=True)

                ndim = len(l_bound)

                nwalkers = 2 * ndim
                
                ''' 
                Block for saving the process, so that the chain doesn't have to be restarted after it has finished.
                This is for cases where convergence has not yet occurred, for example, if after 10000 steps the method still
                has not converged, you can run it again and it will start where it left off, from step 10000.

                NOTE: If you are working with different groups, you should create an .h5 file for each group,
                as in this example. If you don't have separate groups, just remove the repeating structure and leave only
                filename = reinitiate{model_extension}.h5
                '''

                if group == "io_sensitive":
                    filename = f"./reinitiate_files/reinitiate_{group}{model_extension}.h5"
                else:
                    continue

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

                    pos = None

                additional_chain_size = 100         

                print(f'Calibrating {group}, using model {model_extension.split("_")[-1]}')
            
                n_cores = cpu_count()
                print(f'Calibration using {n_cores} cores')
                with Pool(processes = n_cores) as pool:
                    sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability, 
                        args=(l_bound, u_bound, model_extension, full_data, group),
                        pool = pool,
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
                total_size = ((additional_chain_size-additional_chain_size//4)*ndim*2)
                goal_size = 10000
                scale_corner = max(1, total_size // goal_size)
                save_chain = sampler.get_chain(discard=additional_chain_size // 4, flat=True, thin = scale_corner)

                np.savetxt(f'./Output_Calibration/multi_chain_{group}{model_extension}.gz', save_chain)
                np.savez(f'./Output_Calibration/multi_ll_pars_{group}{model_extension}.npz', max_ll=max(flat_ll), pars=best_pars)

                # corner_chain = sampler.get_chain(discard=additional_chain_size // 2, flat=True)
                # fig = corner.corner(corner_chain, labels=labels, truths = best_pars)
                # plt.savefig('./Output_Calibration/multi_corner_' + group + model_extension + '.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                # plt.close()

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
                plt.savefig('./Output_Calibration/multi_chain_' + group + model_extension + '.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                plt.close()

                plot_maxll_solution_global('./Output_Calibration/multi_ll_pars_' + group + model_extension + '.npz', full_data, show = False, nCols=4, figure_name='./Output_Calibration/multi_max_ll_' + group + model_extension)
