import subprocess
import sys
import os
import pandas as pd
import glob
import math
import emcee #
import string 
import corner #
import numpy as np #
from tqdm import tqdm #
from numba import jit #
import concurrent.futures
from multiprocessing import Pool #
from multiprocessing import cpu_count
import matplotlib.pyplot as plt #
import matplotlib.lines as mlines #
import matplotlib.ticker as mticker #
from scipy.integrate import odeint #
import seaborn as sns #
import scipy.stats as stats #
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
class ScalarFormatterWithDecimals(ScalarFormatter):
    def _set_format(self):
        self.format = '%.1f'
import arviz as az

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
    t_days = np.array([int(t) for t in file.split('/')[-1].split('_c')[-1].split('.txt')[0].split('_t')[1:]])
    
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
def model_exp(y, t, growth_rate):
    tumor = y
    tumorVolume = growth_rate * tumor
    return tumorVolume

@jit(nopython=True)
def model_bert(y, t, growth_rate,b):
    tumor = y
    tumorVolume = growth_rate * (tumor**(2./3.)) - b * tumor
    return tumorVolume

@jit(nopython=True)
def model_mendel(y, t, growth_rate, power):
    tumor = y
    tumorVolume = growth_rate * (tumor**power)
    return tumorVolume

@jit(nopython=True)
def model_log(y, t, growth_rate, carrying_capacity):
    tumor = y
    tumorVolume = growth_rate * tumor * (1 - tumor/carrying_capacity)
    return tumorVolume

@jit(nopython=True)
def model_lin(y, t, growth_rate, carrying_capacity):
    tumor = y
    tumorVolume = growth_rate * tumor / (tumor + carrying_capacity)
    return tumorVolume

@jit(nopython=True)
def model_gomp(y, t, growth_rate, b,c):
    tumor = y
    tumorVolume = growth_rate * tumor * np.log(b/(tumor+c))
    return tumorVolume

@jit(nopython=True)
def model_surf(y, t, growth_rate, carrying_capacity):
    tumor = y
    tumorVolume = growth_rate * tumor / ((tumor + carrying_capacity)**(1./3.))
    return tumorVolume


def solve_model(model_extension, time_array, parameters, initial_condition, type_sol = 'data'):
    model_name = 'model' + model_extension
    model_func = globals()[model_name]  # Get the function object by its name
    rhs = getattr(model_func, 'py_func', model_func)
    if type_sol == 'smooth':
        bgn_p = round(time_array[0],1)
        end_p = round(time_array[-1],1)
        time_array = np.linspace(bgn_p,end_p, int((end_p-bgn_p)/0.1)+1)
        sol = odeint(rhs, t = time_array, y0 = [initial_condition], args = parameters, mxstep=2000)
        return np.column_stack((time_array, sol))
    else:
        return odeint(rhs, t = time_array, y0 = [initial_condition], args = parameters, mxstep=2000)


def log_likelihood(theta, model_extension, full_data, group):
 
    ll = 0

    variance = theta[-1] ** 2

    ndim = len(full_data[group])
    for idx,mouse_data in enumerate(full_data[group]):

        time_array = mouse_data['data'][:, 0]

        parameters = tuple(theta[:-(ndim+1)])

        initial_condition = theta[-(ndim+1)+idx]

        solution = solve_model(model_extension, time_array, parameters, initial_condition)

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

def plot_scatter_ax(ax, final_exp, final_model, formatter, ccc_val, ccc_ic, pcc_val, pcc_ic):
    all_cccs = []
    for i in range(len(final_exp)):
        ccc_value = ccc(final_model[i], final_exp[i])
        all_cccs.append(ccc_value)
        ax.plot(final_exp[i], final_model[i], 'o', color='blue')

    ax.set_xlabel(r'Data - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)
    ax.set_ylabel(r'Model - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)

    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)

    final_exp = np.concatenate(final_exp)
    final_model = np.concatenate(final_model)
    max_value = math.ceil(max(max(final_exp), max(final_model)) / 1000) * 1000
    ticks = np.linspace(0, max_value, num=int(max_value/1000)+1)

    ax.set_xlim((0, max_value))
    ax.set_ylim((0, max_value))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    formatter = ScalarFormatterWithDecimals(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 2))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 2), useOffset=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    ax.tick_params(axis='both', which='major', length=6, width=1, direction='out', color='black', bottom=True, left=True)
    ax.minorticks_off()

    pccT = pcc(final_model, final_exp)
    cccT = ccc(final_model, final_exp)
    # mapeT = mape(final_model, final_exp)

    subplot_index = getattr(ax, 'get_subplotspec', lambda: None)()
    label_prefix = ''
    if subplot_index is not None:
        label_prefix = 'A) ' if subplot_index.colspan.start == 0 else 'B) '

    if pcc_ic is not None and ccc_ic is not None:
        # CCC
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
    
    return 

def summarize_posterior(arr, alpha=0.05):
    arr = np.asarray(arr, dtype=float)
    med = np.median(arr)
    lo, hi = np.percentile(arr, [100*alpha/2, 100*(1-alpha/2)])
    hdi_lo, hdi_hi = az.hdi(arr, hdi_prob=1-alpha)
    return med, lo, hi, hdi_lo, hdi_hi

def posterior_corr_from_chain(chain_path, model_extension, full_data, group, n_samp=1000, seed=42):

    if not os.path.exists(chain_path):
        print(f"[ERROR] Chain not found: {chain_path}")
        return None

    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)

    if group not in full_data:
        print(f"[ERROR] Group '{group}' not found in full_data.")
        return None
    ndim = len(full_data[group])

    n_samp = min(n_samp, chain.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(chain.shape[0], size=n_samp, replace=False)
    thetas = chain[idx, :]

    pcc_samples = []
    ccc_samples = []

    for k, theta in enumerate(thetas):
        try:

            params = tuple(theta[:-(ndim + 1)])
            ics    = theta[-(ndim + 1):-1] 

            full_model = []
            full_exp   = []

            for i, mouse in enumerate(full_data[group]):
                t = mouse['data'][:, 0]
                y = mouse['data'][:, 1]
                ic = ics[i]

                sol = solve_model(model_extension, t, params, ic, type_sol='data')

                if sol.ndim == 2 and sol.shape[1] == 1:
                    sol = sol[:, 0]

                full_model.append(sol)
                full_exp.append(y)

            model_concat = np.concatenate(full_model)
            exp_concat   = np.concatenate(full_exp)

            pcc_val = pcc(model_concat, exp_concat)
            ccc_val = ccc(model_concat, exp_concat)

            pcc_val = float(np.clip(pcc_val, -1.0, 1.0))
            ccc_val = float(np.clip(ccc_val, -1.0, 1.0))

            pcc_samples.append(pcc_val)
            ccc_samples.append(ccc_val)
        except Exception as e:
            print(f"[WARN] sample failure {k}: {e}")

    pcc_samples = np.asarray(pcc_samples, dtype=float)
    ccc_samples = np.asarray(ccc_samples, dtype=float)

    pcc_med, pcc_lo, pcc_hi, pcc_hdi_lo, pcc_hdi_hi = summarize_posterior(pcc_samples, alpha=0.05)
    ccc_med, ccc_lo, ccc_hi, ccc_hdi_lo, ccc_hdi_hi = summarize_posterior(ccc_samples, alpha=0.05)

    print(f"\n[Posterior {group} | {model_extension}]  n={len(pcc_samples)} (valid samples)")
    print(f"PCC: median={pcc_med:.3f}, IC95%=[{pcc_lo:.3f}, {pcc_hi:.3f}], HDI95%=[{pcc_hdi_lo:.3f}, {pcc_hdi_hi:.3f}]")
    print(f"CCC: median={ccc_med:.3f}, IC95%=[{ccc_lo:.3f}, {ccc_hi:.3f}], HDI95%=[{ccc_hdi_lo:.3f}, {ccc_hdi_hi:.3f}]")

    return {
        "pcc_samples": pcc_samples,
        "ccc_samples": ccc_samples,
        "pcc_summary": (pcc_med, pcc_lo, pcc_hi, pcc_hdi_lo, pcc_hdi_hi),
        "ccc_summary": (ccc_med, ccc_lo, ccc_hi, ccc_hdi_lo, ccc_hdi_hi)
    }

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

def configure_plot_settings(fontsize):
    plt.rcParams['font.size'] = fontsize
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))
    return formatter

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

    scatter_pairs = {}
    for group, best_chain in best_chain_per_group.items():
        print(f"Processando: Grupo = {group}, Melhor ajuste encontrado com max_ll = {max_ll_per_group[group]}")

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
        scale=1
        print(f'Size pars = {len(theta)}, animals = {ndim}')
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

            theta = npzfile['pars']
            
            parameters = tuple(theta[:-(ndim+1)*scale])
            initial_condition = theta[-(ndim+1)+idx]

            solution = solve_model(model_extension, data[:, 0], parameters, initial_condition, "smooth")

            mask = np.isin(solution[:, 0], data[:, 0])
            matched_solution_times = solution[mask, 0]
            matched_solution_volumes = solution[mask, 1]

            ax.plot(matched_solution_times, matched_solution_volumes, color='black', linestyle='dashed', linewidth=2, label='Model')

            full_model.append(matched_solution_volumes)
            full_exp.append(data[:, 1])

            pccT = pcc(matched_solution_volumes, data[:, 1])
            cccT = ccc(matched_solution_volumes, data[:, 1])

            ax.text(0.5, .05, 'CCC/PCC = {:.2f}/{:.2f}'.format(cccT, pccT), horizontalalignment='left', transform=ax.transAxes)
        
        finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, figure_name, formatter)

        scatter_pairs[group] = (full_exp.copy(), full_model.copy())

    if 'control_sensitive' in scatter_pairs and 'control_resistant' in scatter_pairs:
        (exp_s, mod_s) = scatter_pairs['control_sensitive']
        (exp_r, mod_r) = scatter_pairs['control_resistant']

        fig_sc, axes_sc = plt.subplots(1, 2, figsize=(22, 15 * (1.4 / (1 + 1 + 1.4))))

        res_cs = posterior_corr_from_chain(
            chain_path='./Output_Calibration/multi_chain_control_sensitive'+ model_extension +'.gz',
            model_extension=model_extension,
            full_data=full_data,
            group='control_sensitive',
            n_samp=1000,
            seed=42
        )

        res_cr = posterior_corr_from_chain(
            chain_path='./Output_Calibration/multi_chain_control_resistant'+ model_extension +'.gz',
            model_extension=model_extension,
            full_data=full_data,
            group='control_resistant',
            n_samp=1000,
            seed=42
        )

        ccc_med_cs, ccc_lo_cs, ccc_hi_cs = res_cs["ccc_summary"][:3]
        pcc_med_cs, pcc_lo_cs, pcc_hi_cs = res_cs["pcc_summary"][:3]


        plot_scatter_ax(axes_sc[0], exp_s, mod_s, formatter,
                ccc_val=ccc_med_cs, ccc_ic=[ccc_lo_cs, ccc_hi_cs],
                pcc_val=pcc_med_cs, pcc_ic=[pcc_lo_cs, pcc_hi_cs])

        ccc_med_cr, ccc_lo_cr, ccc_hi_cr = res_cr["ccc_summary"][:3]
        pcc_med_cr, pcc_lo_cr, pcc_hi_cr = res_cr["pcc_summary"][:3]

        plot_scatter_ax(axes_sc[1], exp_r, mod_r, formatter,
                ccc_val=ccc_med_cr, ccc_ic=[ccc_lo_cr, ccc_hi_cr],
                pcc_val=pcc_med_cr, pcc_ic=[pcc_lo_cr, pcc_hi_cr])


        fig_sc.tight_layout()
        if save:
            fig_sc.savefig('./Output_Calibration/multi_scatter_both'+ model_extension +'.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
            fig_sc.savefig('./Output_Calibration/multi_scatter_both'+ model_extension +'.pdf', bbox_inches='tight', pad_inches=0.02)
        if show:
            plt.show()
        else:
            plt.close(fig_sc)

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
    if model_extension == '_exp':
        labels = ["r"]
        l_bound = [0.0] # Lower bounds
        u_bound = [1.0] # Upper bounds
    elif model_extension == '_mendel':
        labels = ["r", "b"]
        l_bound = [1e-6, 1e-6] # Lower bounds
        u_bound = [1.0, 1.0] # Upper bounds
    elif model_extension == '_bert':
        labels = ["r", "b"]
        l_bound = [1e-6, 1e-6] # Lower bounds
        u_bound = [2.0, 1.0] # Upper bounds
    elif model_extension == '_lin':
        labels = ["r", "b"]
        l_bound = [0.0, 0.0000] # Lower bounds
        u_bound = [1.0, ic_max] # Upper bounds
    elif model_extension == '_gomp':
        labels = ["r", "b", "c"]
        l_bound = [1e-6, 1e-6, 1e-6] # Lower bounds
        u_bound = [1.0, 10*ic_max, ic_max] # Upper bounds
    elif model_extension == '_surf':
        labels = ["r", "b"]
        l_bound = [1e-6, 1e-6] # Lower bounds
        u_bound = [1.0, ic_max] # Upper bounds
    elif model_extension == '_log':
        labels = ["r", "cc"]
        l_bound = [0.0, 100.] # Lower bounds
        u_bound = [1.0, 20000] # Upper bounds
    if bool_multiple_mice:       
        for i in range(group_size):
            labels.append("ic" + str(i+1))
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

if __name__ == "__main__":
# Find and sort all text files in the ./data directory
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
        t_days = np.array([int(t) for t in file.split('/')[-1].split('_c')[-1].split('.txt')[0].split('_t')[1:]])
        
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
            'treatment_days': t_days
        })

    run_calibration = True
    if run_calibration:
        for model_extension in [
                                '_exp', 
                                '_mendel',
                                '_bert', 
                                '_lin', 
                                '_gomp', 
                                '_surf', 
                                '_log'
                                ]:
            print(f"Solving model {model_extension}")
            for group in ['control_sensitive', 'control_resistant']:
                print(f"Solving group {group}")

                group_size = len(full_data[group])
                
                l_bound, u_bound, labels = define_bounds_labels(model_extension, bool_multiple_mice=True)

                ndim = len(l_bound)            

                nwalkers = 2 * ndim            

                if group == "control_sensitive":
                    filename = f"./reinitiate_files/reinitiate_{group}{model_extension}.h5"
                else:
                    filename = f"./reinitiate_files/reinitiate_{group}{model_extension}.h5"

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

                additional_chain_size = 1000

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

                corner_chain = sampler.get_chain(discard=additional_chain_size // 2, flat=True)
                fig = corner.corner(corner_chain, labels=labels)
                plt.savefig('./Output_Calibration/multi_corner_' + group + model_extension + '.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
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
                plt.savefig('./Output_Calibration/multi_chain_' + group + model_extension + '.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                plt.close()

                plot_maxll_solution_global('./Output_Calibration/multi_ll_pars_' + group + model_extension + '.npz', full_data, show = False ,nCols=4, figure_name='./Output_Calibration/multi_max_ll_' + group + model_extension)

            plot_maxll_solution_global(f'./Output_Calibration/multi_ll_pars_*{model_extension}.npz',
                           full_data, show=False, nCols=4,
                           figure_name=f'./Output_Calibration/multi_max_ll_both{model_extension}')
            
            res_cs = posterior_corr_from_chain(
            chain_path='./Output_Calibration/multi_chain_control_sensitive' + model_extension + '.gz',
            model_extension=model_extension,
            full_data=full_data,
            group='control_sensitive',
            n_samp=1000,
            seed=42
            )

            res_cr = posterior_corr_from_chain(
                chain_path='./Output_Calibration/multi_chain_control_resistant' + model_extension + '.gz',
                model_extension=model_extension,
                full_data=full_data,
                group='control_resistant',
                n_samp=1000,
                seed=42
            )
