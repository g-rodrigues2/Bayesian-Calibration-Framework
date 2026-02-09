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
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
class ScalarFormatterWithDecimals(ScalarFormatter):
    def _set_format(self):
        self.format = '%.1f'


# Find and sort all text files in the ./data directory
data_files = sorted(glob.glob("./data/*.txt"))

# Dictionary to store the parsed data, grouped by data type
full_data = {}

# Loop through each file in the data_files list
for file in data_files:
   
    base_name = os.path.basename(file).split(".txt")[0]

    data_type = base_name.split("_c")[0]

    mouse_name = "c" + base_name.split("_c")[-1].split("_t")[0]

    no_suv_str = base_name.split("_SUV")[0]  

    t_days_str = no_suv_str.split("_c")[-1]

    tau_radiation = np.array([0, 1, 2, 3, 4, 5])
    tau_immuno = np.array([0, 3, 6, 9, 12, 15, 18])
    tau_immuno_post_radiation = np.array([9, 12, 15, 18, 21, 24])

    suv_value = float(base_name.split("_SUV")[-1])

    data = np.loadtxt(file)

    treatment = data_type 

    if data_type not in full_data:
        full_data[data_type] = []

    full_data[data_type].append({
        'name': mouse_name,
        'data': data,  
        'treatment': treatment,
        'treatment_days': {
            'radiation': tau_radiation,
            'immuno': tau_immuno,
            'rad_immuno': tau_immuno_post_radiation
        },
        'SUV': suv_value
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
def model_RadImmuno(
    y, t,
    growth_rate, carrying_capacity,
    a_radiation, b_radiation, delay,
    a, b, SUV, s_immuno, lam,             
    tau_radiation, tau_immuno,
    dose_schedule_rad, dose_schedule_immuno,
):

    tumor = y[0]
    if tumor < 1e-12:
        tumor = 1e-12

    tumor_growth = growth_rate * tumor * (1.0 - tumor / carrying_capacity)

    treatment_rad_effect = 0.0
    n_rad = dose_schedule_rad.shape[0]
    for i in range(n_rad):
        dose = dose_schedule_rad[i]
        tau  = tau_radiation[i]
        if t >= tau + delay:
            treatment_rad_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau - delay))

    T_pow   = tumor ** lam
    denom_T = s_immuno * T_pow

    treatment_immuno_effect = 0.0
    n_imm = dose_schedule_immuno.shape[0]
    for i in range(n_imm):
        dose = dose_schedule_immuno[i]
        tau  = tau_immuno[i]
        if t >= tau:
            dose_pow = dose ** lam
            denom = denom_T + dose_pow
            if denom > 0.0:
                phi = (a * SUV + b) 
                treatment_immuno_effect +=  phi * (dose_pow / denom)

    dTdt = tumor_growth - (treatment_rad_effect + treatment_immuno_effect) * tumor

    return dTdt

def _to_float_array_or_empty(x):

    return np.asarray(x, dtype=np.float64) if x is not None else np.empty(0, dtype=np.float64)

def solve_model(model_extension, time_array, parameters, initial_condition, treatment_days_rad=None, dose_rad=None, treatment_days_immuno=None, dose_immuno=None, type_sol='data'):

    model_name = 'model' + model_extension
    model_func = globals()[model_name]

    tau_radiation = treatment_days_rad
    tau_immuno = treatment_days_immuno

    tau_radiation = _to_float_array_or_empty(treatment_days_rad)
    tau_immuno    = _to_float_array_or_empty(treatment_days_immuno)

    dose_schedule_rad    = np.full(tau_radiation.shape[0], dose_rad, dtype=np.float64)
    dose_schedule_immuno = np.full(tau_immuno.shape[0],    dose_immuno, dtype=np.float64)

    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p - bgn_p) / 0.1) + 1)
        sol = odeint(
            model_func, 
            t=time_array, 
            y0=[initial_condition], 
            args=(*parameters, tau_radiation, tau_immuno, dose_schedule_rad, dose_schedule_immuno), 
            mxstep=2000
        )
        return np.column_stack((time_array, sol))
    else:        
        return  odeint(
                model_func, 
                t=time_array, 
                y0=[initial_condition], 
                args=(*parameters, tau_radiation, tau_immuno, dose_schedule_rad, dose_schedule_immuno), 
                mxstep=2000
                )


def log_likelihood(theta, model_extension, full_data, group):
 
    ll = 0
    growth_rate = 8.89429860e-02  
    carrying_capacity = 3.69671552e+03 

    s = 0.015152682754421672
    lam = 1.928733156897561

    variance = theta[-1] ** 2

    ndim = len(full_data[group])

    for idx, mouse_data in enumerate(full_data[group]):
        SUV = float(mouse_data['SUV'])

 
        dose_rad = 2.0  
        dose_immuno = 1.0  
 
        a_radiation = 1.52716650e-02
        b_radiation = 6.91980858e-02
        delay = 4.76117778e+00
        td_rad    = mouse_data['treatment_days']['radiation']
        td_immuno = mouse_data['treatment_days']['rad_immuno'] 

        time_array = mouse_data['data'][:, 0]

        suv_func_params = theta[:-(ndim + 1)]  

        initial_condition = theta[-(ndim + 1) + idx]

        solution = solve_model(
            model_extension,
            time_array,
            (growth_rate, carrying_capacity, a_radiation, b_radiation, delay, *suv_func_params, SUV, s, lam),
            initial_condition,
            td_rad,
            dose_rad,
            td_immuno,
            dose_immuno
        )

        observed_volume = mouse_data['data'][:, 1] 

        ll += -0.5 * np.sum((solution[:,0] - observed_volume) ** 2 / variance + np.log(2 * np.pi) + np.log(variance))

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

def extract_ab_std_from_theta(theta):
    a = float(theta[0])
    b = float(theta[1])
    std = float(theta[-1])
    return a, b, std

def log_likelihood_ic_only(theta, model_extension, mouse_data, a_fixed, b_fixed):

    ic0, std = theta[0], theta[1]
    if std <= 0:
        return -np.inf

    growth_rate = 8.89429860e-02
    carrying_capacity = 3.69671552e+03
    s = 0.015152682754421672
    lam = 1.928733156897561
    dose_rad = 2.0
    dose_immuno = 1.0

    SUV = float(mouse_data['SUV'])
    data = mouse_data['data']
    time_array = data[:,0]
    obs = data[:,1]

    a_radiation = 1.52716650e-02
    b_radiation = 6.91980858e-02
    delay = 4.76117778e+00
    td_rad = mouse_data['treatment_days']['radiation']
    td_immuno = mouse_data['treatment_days']['rad_immuno']

    sol = solve_model(
        model_extension,
        time_array,
        (growth_rate, carrying_capacity, a_radiation, b_radiation, delay,
         a_fixed, b_fixed, SUV, s, lam),
        ic0, td_rad, dose_rad, td_immuno, dose_immuno, type_sol='data'
    ).ravel()

    var = std**2
    return -0.5*np.sum((sol - obs)**2/var + np.log(2*np.pi) + np.log(var))


def log_prior_ic_only(theta, ic_lower, ic_upper, std_lower, std_upper):
    ic0, std = theta[0], theta[1]
    if not (ic_lower < ic0 < ic_upper): return -np.inf
    if not (std_lower < std < std_upper): return -np.inf
    return 0.0


def log_probability_ic_only(theta, ic_lower, ic_upper, std_lower, std_upper,
                            model_extension, mouse_data, a_fixed, b_fixed):
    lp = log_prior_ic_only(theta, ic_lower, ic_upper, std_lower, std_upper)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ic_only(theta, model_extension, mouse_data, a_fixed, b_fixed)


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

def plot_scatter_new_version(final_exp, final_model, ax, save, show, figure_name, formatter):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    c = 1
    all_cccs = []
    for i in range(len(final_exp)):
        ccc_val = ccc(final_model[i], final_exp[i])
        all_cccs.append(ccc_val)
        if ccc_val < 0.8:
            ax.plot(final_exp[i], final_model[i], 'o', color='blue')
            c += 1
        else:
            ax.plot(final_exp[i], final_model[i], 'o', color='blue')
    ax.set_xlabel(r'Data - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)
    ax.set_ylabel(r'Model - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)
    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)
    final_exp_concat = np.concatenate(final_exp)
    final_model_concat = np.concatenate(final_model)
    max_value = max((max(final_exp_concat), max(final_model_concat)))
    max_value = math.ceil(max_value / 1000) * 1000
    ticks = np.linspace(0, max_value, num=int(max_value/1000)+1)
    
    ax.set_ylim((0, max_value))
    ax.set_xlim((0, max_value))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


    formatter = ScalarFormatterWithDecimals(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 2))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 2), useOffset=False)
    ax.grid(False)
    ax.yaxis.offsetText.set_visible(False) 
    ax.xaxis.offsetText.set_visible(False) 

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=6,
        width=1,
        colors='black',
        bottom=True,
        top=False,
        left=True,
        right=False
    )

    pccT = pcc(final_model_concat, final_exp_concat)
    cccT = ccc(final_model_concat, final_exp_concat)

    all_cccs = np.array(all_cccs)
    ax.text(0.02, .9, 'CCC/PCC = {:.2f}/{:.2f}'.format(cccT, pccT, all_cccs.mean()),
            horizontalalignment='left', transform=ax.transAxes, fontsize=18)
    return


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
    #ax.text(0.025, .8, 'CCC/PCC = {:.2f}/{:.2f}\n CCC = {:.2f}$\pm${:.2f}'.format(cccT, pccT, all_cccs.mean(), all_cccs.std()), horizontalalignment='left', transform=ax.transAxes)
    
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

    outdir = './Output_Calibration'
    os.makedirs(outdir, exist_ok=True)
    saved_excel_paths = {} 
    
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

    s = 0.015152682754421672
    lam = 1.928733156897561

    for group, best_chain in best_chain_per_group.items():
        print(f"Processing: Group = {group}, Best fit found with max_ll = {max_ll_per_group[group]}")

        npzfile = np.load(best_chain)
        theta = npzfile['pars']

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
            SUV = float(mouse_data['SUV'])
            mouse_name = mouse_data['name']
            print(f"[DEBUG] Processing mouse: {mouse_name}")
            ax = axes[idx]

            data = mouse_data['data']

            dose_rad = 2.0 
            dose_immuno = 1.0 

            suv_func_params = theta[:-(ndim + 1)] 
            initial_condition = theta[-(ndim + 1) + idx]

            ax.set_ylabel('Tumor Volume (mm³)')
            ax.text(0.05, .9, f'{string.ascii_uppercase[idx]}) {mouse_name}', horizontalalignment='left', transform=ax.transAxes, fontsize=18)

            ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10, label='Tumor Volume')
            ax.set_xlabel('Time (days)')
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim((-1, max_time_group + 1))


            a_radiation = 1.52716650e-02
            b_radiation = 6.91980858e-02
            delay = 4.76117778e+00
            td_rad    = mouse_data['treatment_days']['radiation']
            td_immuno = mouse_data['treatment_days']['rad_immuno']     

            for i, t_day in enumerate(td_rad):
                ax.axvline(
                    x=t_day,
                    color='green',
                    linestyle='--',
                    linewidth=1,
                    label='Radiotherapy' if i==0 else ""
                )
            for i, t_day in enumerate(td_immuno):
                ax.axvline(
                    x=t_day,
                    color='red',
                    linestyle='--',
                    linewidth=1,
                    label='Immunotherapy' if i==0 else ""
                )
    
            solution = solve_model(
                model_extension, 
                data[:, 0], 
                (growth_rate, carrying_capacity, a_radiation, b_radiation, delay, *suv_func_params, SUV, s, lam),
                initial_condition, 
                td_rad, 
                dose_rad,
                td_immuno, 
                dose_immuno,
                type_sol='smooth'
            )

            mask = np.isin(solution[:, 0], data[:, 0])
            matched_solution_times = solution[mask, 0]
            matched_solution_volumes = solution[mask, 1]

            ax.plot(solution[:, 0], solution[:, 1], color='black', linestyle='dashed', linewidth=2, label='Model')

            full_model.append(matched_solution_volumes)
            full_exp.append(data[:, 1])

            pccT = pcc(matched_solution_volumes, data[:, 1])
            cccT = ccc(matched_solution_volumes, data[:, 1])
            mapeT = mape(matched_solution_volumes, data[:, 1])

            ax.text(0.5, .05, 'CCC/PCC/MAPE = {:.2f}/{:.2f}/{:.2f}%'.format(cccT, pccT, mapeT), horizontalalignment='left', transform=ax.transAxes)

        finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, figure_name, formatter)
  
        fig_sc, ax_sc = plt.subplots(figsize=(13,7))
        plot_scatter_new_version(
            full_exp,
            full_model,
            ax_sc,
            save=True,
            show=False,
            figure_name='./Output_Calibration/multi_max_ll_rad+io_sensitive_'+ model_extension +'_sp',
            formatter=formatter
        )
        fig_sc.savefig('./Output_Calibration/multi_max_ll_rad+io_sensitive_'+ model_extension +'_sp.png', dpi=300, bbox_inches='tight')

        
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

    if model_extension == '_RadImmuno':
        labels = ["a", "b"]
        l_bound = [-10.0, -10.0]
        u_bound = [10.0, 10.0]

    # Se tivermos vários camundongos
    if bool_multiple_mice:
        # 1) Condições iniciais (IC) para cada camundongo
        for i in range(group_size):
            labels.append(f"ic{i+1}")
            l_bound.append(0.0000)
            u_bound.append(ic_max)

        # 3) std
        labels.append("std")
        l_bound.append(0.1)
        u_bound.append(300.0)

    else:
        # Caso single mouse (não mostrado, mas mantemos a lógica)
        labels.append("ic")
        l_bound.append(0.0000)
        u_bound.append(ic_max)
        labels.append("std")
        l_bound.append(0.1)
        u_bound.append(1000.0)

    return l_bound, u_bound, labels


def calibrate_ic_bayesian_for_mouse(mouse_data, a_fixed, b_fixed, model_extension,
                                    left_out, group,
                                    steps=15000, burn=2000): 
    ic_lower, ic_upper = 0.0, max_ic*2
    std_lower, std_upper = 0.1, 300.0

    ndim = 2
    nwalkers = 2*ndim 

    os.makedirs("./reinitiate_files", exist_ok=True)
    filename = f"./reinitiate_files/reinitiate_{group}_PRED-{left_out}{model_extension}.h5"
    if not os.path.exists(filename):
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        pos = np.zeros((nwalkers, ndim))
        for w in range(nwalkers):
            pos[w,0] = np.random.uniform(low=ic_lower,  high=ic_upper)  
            pos[w,1] = np.random.uniform(low=std_lower, high=std_upper) 
    else:
        backend = emcee.backends.HDFBackend(filename)
        pos = None

    with Pool(processes=cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_ic_only,
            args=(ic_lower, ic_upper, std_lower, std_upper,
                  model_extension, mouse_data, a_fixed, b_fixed),
            pool=pool, backend=backend
        )
        if backend.iteration > 0:
            sampler.run_mcmc(None, steps, progress=True)
        else:
            sampler.run_mcmc(pos, steps, progress=True)

    chain = sampler.get_chain(discard=burn, flat=True)
    logp  = sampler.get_log_prob(discard=burn, flat=True)

    if chain.size == 0 or logp.size == 0:
        print(f"[WARN] Empty chain after burn for {left_out}. "
              f"Try increasing steps or decreasing burn. steps={steps}, burn={burn}")

        chain = sampler.get_chain(flat=True)
        logp  = sampler.get_log_prob(flat=True)

    best_idx = np.argmax(logp)
    ic_hat, std_hat = chain[best_idx]

    os.makedirs('./Output_Calibration', exist_ok=True)
 
    thin =  max(1, (steps-burn)//10000)
    save_chain = sampler.get_chain(discard=burn, flat=True, thin=thin)

    np.savetxt(f'./Output_Calibration/loo_ic_chain_{group}_PRED-{left_out}{model_extension}.gz',
               save_chain)
    np.savez(f'./Output_Calibration/loo_ic_pars_{group}_PRED-{left_out}{model_extension}.npz',
             max_ll=np.max(logp),
             pars=np.array([ic_hat, std_hat, a_fixed, b_fixed]))

    fig, axes = plt.subplots(nrows=ndim, ncols=1, figsize=(10, 2*ndim), sharex=True)
    samples = sampler.get_chain()
    labels = ['IC', 'std']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i]); ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.savefig(f'./Output_Calibration/loo_ic_chain_{group}_PRED-{left_out}{model_extension}.png',
                format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    return float(ic_hat), float(std_hat)


def predict_all_leftouts_and_plot(npz_glob, full_data, group, model_extension,
                                  nCols=4,
                                  fig_name_prefix='./Output_Calibration/loo_pred_' ):

    full_exp = []
    full_model = []
    mice_order = []

    mice_by_name = {m['name']: m for m in full_data[group]}
    files = sorted(glob.glob(npz_glob))
    if not files:
        print(f"[WARN] No npz found in: {npz_glob}")
        return

    max_times = find_max_time_per_group(full_data)
    max_time_group = max_times[group]

    nScenarios = len(files)
    nRows = int(np.ceil(nScenarios / nCols))
    formatter = mticker.ScalarFormatter(useMathText=True); formatter.set_powerlimits((-3,2))
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*11, nRows*5))
    axes = axes.ravel()

    growth_rate = 8.89429860e-02
    carrying_capacity = 3.69671552e+03
    s = 0.015152682754421672
    lam = 1.928733156897561
    dose_rad = 2.0
    dose_immuno = 1.0

    for k, npz_path in enumerate(files):
        base = os.path.basename(npz_path)
        try:
            left_out = base.split('_EXCL-')[1].split(model_extension)[0]
        except:
            print(f"[SKIP] I couldn't infer the name of the mouse from {base}.")
            continue

        print(f"\n[INFO] Running Bayesian calibration for mouse ‘{left_out}’ from group '{group}'")  # 👈 AQUI

        if left_out not in mice_by_name:
            print(f"[SKIP] {left_out} is not in {group}")
            continue

        mouse_data = mice_by_name[left_out]

        npzfile = np.load(npz_path)
        theta = npzfile['pars']
        a_fixed, b_fixed, _std_from_group = extract_ab_std_from_theta(theta)

        ic_hat, std_hat = calibrate_ic_bayesian_for_mouse(
            mouse_data, a_fixed, b_fixed, model_extension, left_out, group,
            steps=15000, burn=2000
        )

        SUV = float(mouse_data['SUV'])
        data = mouse_data['data']
        times = data[:,0]
        obs = data[:,1]

        a_radiation = 1.52716650e-02
        b_radiation = 6.91980858e-02
        delay = 4.76117778e+00
        td_rad = mouse_data['treatment_days']['radiation']
        td_immuno = mouse_data['treatment_days']['rad_immuno']

        print(
            f"[PRED] group={group} | mouse={left_out} | model={model_extension} | "
            f"a_fixed={a_fixed:.6g}, b_fixed={b_fixed:.6g}, "
            f"IC_hat={ic_hat:.6g}, STD_hat={std_hat:.6g}, SUV={SUV:.6g} | "
            f"r={growth_rate:.6g}, K={carrying_capacity:.6g} | "
            f"rad(a={a_radiation:.6g}, b={b_radiation:.6g}, delay={delay:.6g}), "
            f"dose_rad={dose_rad}, dose_immuno={dose_immuno} | "
        )

        sol = solve_model(
            model_extension,
            times,
            (growth_rate, carrying_capacity, a_radiation, b_radiation, delay,
             a_fixed, b_fixed, SUV, s, lam),
            ic_hat, td_rad, dose_rad, td_immuno, dose_immuno, type_sol='smooth'
        )


        mask = np.isin(sol[:,0], times)
        yhat = sol[mask,1]

        full_exp.append(obs)
        full_model.append(yhat)
        mice_order.append(left_out)

        ax = axes[k]
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.set_xlabel('Time (days)')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlim((-1, max_time_group + 1))
        ax.plot(times, obs, 'o', color='black', markersize=10, label='Tumor Volume')
        ax.plot(sol[:,0], sol[:,1], color='black', linestyle='dashed', linewidth=2, label='Model')
        ax.text(0.05, .9, f'{string.ascii_uppercase[k]}) {left_out}', transform=ax.transAxes, fontsize=18)

        if td_rad is not None:
            for i,t_day in enumerate(td_rad):
                ax.axvline(x=t_day, color='green', linestyle='--', linewidth=1,
                           label='Radiotherapy' if i==0 else "")
        for i,t_day in enumerate(td_immuno):
            ax.axvline(x=t_day, color='red', linestyle='--', linewidth=1,
                       label='Immunotherapy' if i==0 else "")

        _p = pcc(yhat, obs); _c = ccc(yhat, obs); _m = mape(yhat, obs)
        ax.text(0.5, .05, f'CCC/PCC/MAPE = {_c:.2f}/{_p:.2f}/{_m:.2f}%', transform=ax.transAxes)

    finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model,
                       save=True, show=False,
                       figure_name=f'{fig_name_prefix}{group}{model_extension}',
                       formatter=formatter)

    fig_sc, ax_sc = plt.subplots(figsize=(13,7))
    plot_scatter_new_version(full_exp, full_model, ax_sc,
                             save=True, show=False,
                             figure_name=f'{fig_name_prefix}{group}{model_extension}_sp',
                             formatter=formatter)
    fig_sc.savefig(f'{fig_name_prefix}{group}{model_extension}_sp.png', dpi=300, bbox_inches='tight')
    plt.close(fig_sc)


def make_loo_full_data(full_data, group, leave_out_mouse):
    fd = {g: (list(mice) if g != group else [m for m in mice if m['name'] != leave_out_mouse])
          for g, mice in full_data.items()}
    return fd

if __name__ == "__main__":
            # Find and sort all text files in the ./data directory
        # Find and sort all text files in the ./data directory
    data_files = sorted(glob.glob("./data/*.txt"))

    # Dictionary to store the parsed data, grouped by data type
    full_data = {}

    # Loop through each file in the data_files list
    for file in data_files:
        # Pega o nome do arquivo sem o path e sem a extensão .txt
        base_name = os.path.basename(file).split(".txt")[0]
        # Exemplo: "io_sensitive_c0_m3_t9_t12_t15_t18_t21_t24_SUV0.728827738"
        
        # Extrai o data_type (por exemplo "io_sensitive")
        data_type = base_name.split("_c")[0]
        
        # Extrai o mouse_name (por exemplo "c0_m3")
        mouse_name = "c" + base_name.split("_c")[-1].split("_t")[0]
        
        # Separa a parte do nome que contém '_SUV'
        # para conseguir extrair as t-days e o valor de SUV
        no_suv_str = base_name.split("_SUV")[0]  
        # -> "io_sensitive_c0_m3_t9_t12_t15_t18_t21_t24"
        
        # Agora, isolamos o pedaço após '_c' para ficar só "0_m3_t9_t12_t15_t18_t21_t24"
        t_days_str = no_suv_str.split("_c")[-1]
        
        # Define the treatment days for radiation and immunotherapy
        tau_radiation = np.array([0, 1, 2, 3, 4, 5])
        tau_immuno = np.array([0, 3, 6, 9, 12, 15, 18])
        tau_immuno_post_radiation = np.array([9, 12, 15, 18, 21, 24])
        
        # Por fim, o valor de SUV está logo após '_SUV'
        suv_value = float(base_name.split("_SUV")[-1])
        
        # Carrega o arquivo de dados (por exemplo, tempo e volume tumoral)
        data = np.loadtxt(file)
        
        # Se quiser extrair especificamente o "tipo de tratamento",
        # você pode mantê-lo como data_type ou fazer:
        treatment = data_type  # ou data_type.split('_')[0], etc.
        
        # Se ainda não existe a chave data_type em full_data, criamos
        if data_type not in full_data:
            full_data[data_type] = []
        
        # Adicionamos todas as informações (incluindo SUV) no dicionário
        full_data[data_type].append({
            'name': mouse_name,
            'data': data,  # (colunas do .txt)
            'treatment': treatment,
            'treatment_days': {
                'radiation': tau_radiation,
                'immuno': tau_immuno,
                'rad_immuno': tau_immuno_post_radiation
            },
            'SUV': suv_value
        })

    run_calibration = True
    if run_calibration:
        for model_extension in [
                                '_RadImmuno',

                                ]:

            for group in ['rad+io_sensitive']:
                print(f"Solving group {group}")

                mouse_names = [m['name'] for m in full_data[group]]

                for left_out in mouse_names:
                    print(f"\n[LOO] Excluding mouse: {left_out}")

                    full_data_loo = make_loo_full_data(full_data, group, left_out)

                    group_size = len(full_data_loo[group])

                    l_bound, u_bound, labels = define_bounds_labels(model_extension, bool_multiple_mice=True)

                    ndim = len(l_bound)

                    nwalkers = 2 * ndim

                    filename = f"./reinitiate_files/reinitiate_{group}_EXCL-{left_out}{model_extension}.h5"

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

                    additional_chain_size = 10

                    print(f'[LOO] Calibrating {group} (excl. {left_out}), model {model_extension.split("_")[-1]}')
                    n_cores = cpu_count()
                    print(f'Calibration using {n_cores} cores')

                    with Pool(processes=n_cores) as pool:
                        sampler = emcee.EnsembleSampler(
                            nwalkers, ndim, log_probability,
                            args=(l_bound, u_bound, model_extension, full_data_loo, group),
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

                    total_size = ((additional_chain_size - additional_chain_size//4) * ndim * 2)
                    goal_size = 10000
                    scale_corner = max(1, total_size // goal_size)
                    save_chain = sampler.get_chain(discard=additional_chain_size // 4, flat=True, thin=scale_corner)

                    chain_path = f'./Output_Calibration/multi_chain_{group}_EXCL-{left_out}{model_extension}.gz'
                    npz_path   = f'./Output_Calibration/multi_ll_pars_{group}_EXCL-{left_out}{model_extension}.npz'
                    png_chain  = f'./Output_Calibration/multi_chain_{group}_EXCL-{left_out}{model_extension}.png'

                    np.savetxt(chain_path, save_chain)
                    np.savez(npz_path, max_ll=max(flat_ll), pars=best_pars)

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
                    plt.savefig(png_chain, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                    plt.close()

                    plot_maxll_solution_global(
                        npz_path, 
                        full_data_loo,
                        show=False, nCols=4,
                        figure_name=f'./Output_Calibration/multi_max_ll_{group}_EXCL-{left_out}{model_extension}'
                    )
                
                #### After running all LOOs, we can generate a global summary of LOO predictions for each group/model:

                # npz_glob = f'./Output_Calibration/multi_ll_pars_{group}_EXCL-*{model_extension}.npz'
                
                # predict_all_leftouts_and_plot(
                #     npz_glob=npz_glob,
                #     full_data=full_data,
                #     group=group,
                #     model_extension=model_extension,
                #     nCols=4,
                #     fig_name_prefix='./Output_Calibration/loo_pred_'
                # )

