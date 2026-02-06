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
import sys


def read_data(files_location):
    # Find and sort all text files in the ./data directory
    data_files = sorted(glob.glob(files_location))
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
        
    # Return the dictionary containing all the data
    return full_data


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

@jit(nopython=True)
def model_LQcumulativo(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, tau_radiation, dose_schedule):
    """
    Modelo logístico com efeito da radioterapia (LQ model).
    """
    # Crescimento logístico
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    # Efeito do tratamento (radiação)
    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:  # Verifica se o tratamento foi aplicado
            treatment_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau))

    # Ajuste do volume tumoral pelo tratamento
    tumor_volume -= treatment_effect * tumor  # Redução do tumor pela radioterapia

    return tumor_volume

@jit(nopython=True)
def model_LQ(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, tau_radiation, dose_schedule):
    """
    Modelo logístico de crescimento tumoral com efeito da radioterapia (LQ model).
    """
    # Crescimento logístico
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    # Efeito da radioterapia com o modelo LQ
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau:  # Verifica se o tratamento foi aplicado
            # Sobrevivência celular após radiação (modelo LQ)
            survival_factor = np.exp((-a_radiation * dose - b_radiation * dose**2))
            tumor_volume *= survival_factor  # Ajusta o volume do tumor pelo fator de sobrevivência

    return tumor_volume


def find_max_initial_condition(full_data):
    """
    Função para encontrar a maior condição inicial (IC) entre todos os camundongos de ambos os grupos.

    Parâmetros:
    - full_data: Dicionário contendo os dados de todos os camundongos agrupados por tratamento.

    Retorna:
    - maior_ic: O maior valor de condição inicial (IC) encontrado.
    - grupo_maior_ic: O grupo ao qual pertence o camundongo com a maior condição inicial (Control Sensitive ou Control Resistant).
    - camundongo_maior_ic: O nome do camundongo com a maior condição inicial.
    """
    maior_ic = -float('inf')  # Inicializa com -infinito para garantir que qualquer valor será maior
    grupo_maior_ic = None
    camundongo_maior_ic = None

    # Iterar sobre os grupos
    for group, mice_data in full_data.items():
        # Iterar sobre os dados de cada camundongo no grupo
        for mouse in mice_data:
            # A condição inicial é o primeiro valor de volume tumoral registrado
            ic = mouse['data'][0, 1]  # Primeira linha (tempo inicial), segunda coluna (volume tumoral)

            # Comparar se a condição inicial atual é maior do que a maior já registrada
            if ic > maior_ic:
                maior_ic = ic
                grupo_maior_ic = group
                camundongo_maior_ic = mouse['name']

    return maior_ic, grupo_maior_ic, camundongo_maior_ic

def define_bounds_labels(model_extension, full_data, group_size, bool_multiple_mice = False):
    labels = []
    l_bound = []
    u_bound = []
    max_ic, _, _ = find_max_initial_condition(full_data)
    ic_max = max_ic * 2

    if model_extension == '_exp':
        labels = ["r"]
        l_bound = [0.0] # Lower bounds
        u_bound = [1.0] # Upper bounds
    elif model_extension == '_mendel':
        labels = ["r", "b"]
        l_bound = [0.0, 0.0] # Lower bounds
        u_bound = [1.0, 1.0] # Upper bounds
    elif model_extension == '_bert':
        labels = ["r", "b"]
        l_bound = [0.0, 0.0] # Lower bounds
        u_bound = [2.0, 1.0] # Upper bounds
    elif model_extension == '_lin':
        labels = ["r", "b"]
        l_bound = [0.0, 0.0000] # Lower bounds
        u_bound = [ic_max, ic_max] # Upper bounds
    elif model_extension == '_gomp':
        labels = ["r", "b", "c"]
        l_bound = [0.0, 0.0000, 0.0000] # Lower bounds
        u_bound = [1.0, 10*ic_max, ic_max] # Upper bounds
    elif model_extension == '_surf':
        labels = ["r", "b"]
        l_bound = [0.0, 0.0000] # Lower bounds
        u_bound = [ic_max, ic_max] # Upper bounds
    elif model_extension == '_log':
        labels = ["r", "cc"]
        l_bound = [0.0, 100.] # Lower bounds
        u_bound = [1.0, 20000] # Upper bounds

    elif model_extension == '_LQcumulativo':
        labels = ["alpha_rad", "beta_rad"]  # Alpha, Beta, Tempo de tratamento, Dose aplicada
        l_bound = [0.0, 0.0]  # Limites inferiores para Alpha, Beta, Tempo de tratamento, Dose aplicada
        u_bound = [1, 1]  # Limites superiores para Alpha, Beta, Tempo de tratamento, Dose aplicada
        
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

def find_max_time_per_group(full_data):
    max_times = {}

    # Iterar sobre cada grupo em full_data
    for group, mice_data in full_data.items():
        max_time = 0
        
        # Iterar sobre os dados de cada rato no grupo
        for mouse in mice_data:
            time_data = mouse['data'][:, 0]  # A primeira coluna é o tempo
            max_time_mouse = max(time_data)  # Acha o valor máximo de tempo para o rato atual
            
            # Atualiza o tempo máximo do grupo, se o tempo do rato atual for maior
            if max_time_mouse > max_time:
                max_time = max_time_mouse
        
        # Armazena o valor máximo de tempo para o grupo
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

def plot_scatter(final_exp, final_model, save, show, figure_name, formatter):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(final_exp, final_model, 'o', color='blue')
    ax.set_xlabel('Data - Tumor volume (mm³)')
    ax.set_ylabel('Model - Tumor volume (mm³)')
    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)
    
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
    ax.text(0.025, .85, 'CCC/PCC = {:.2f}/{:.2f}\nMAPE = {:.2f}%'.format(cccT, pccT, mapeT), horizontalalignment='left', transform=ax.transAxes)
    
    if save:
        plt.savefig(figure_name + '_sp.pdf', bbox_inches='tight', pad_inches=0.02)
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












# def include_treatments_in_parameters(parameters, TRAS_treat, TUCA_treat, IMMU_treat, model_extension):
#     if TRAS_treat.size == 0:
#         parameters = parameters[:1] + (0,) + (0,) + (TRAS_treat,) + parameters[1:]
#     else:
#         parameters = parameters[:3] + (TRAS_treat,) + parameters[3:]
#     if TUCA_treat.size == 0:
#         parameters = parameters[:4] + (0,) + (0,) + (TUCA_treat,) + parameters[4:]
#     else:
#         parameters = parameters[:6] + (TUCA_treat,) + parameters[6:]
#     if IMMU_treat.size == 0:
#         parameters = parameters[:7] + (0,) + (0,) + (IMMU_treat,) + parameters[7:]
#     else:
#         parameters = parameters[:9] + (IMMU_treat,) + parameters[9:]
#     if TRAS_treat.size == 0 and model_extension == '_Vexp_uncoupled':
#         parameters = parameters[:12] + (0,) + parameters[12:]
#     return parameters

# def treatment_days(group_number):
#     TRAS_treat = np.array([])
#     TUCA_treat = np.array([])
#     IMMU_treat = np.array([])
#     if group_number == 1 or group_number == 3:
#         IMMU_treat = np.array([0, 3, 6, 9, 12])
#     if group_number == 2 or group_number == 3:
#         TRAS_treat = np.array([0, 3, 6, 9, 12])
#     elif group_number == 4 or group_number == 5 or group_number == 7:
#         TRAS_treat = np.array([0, 4, 8])
#         TUCA_treat = np.array([0, 4, 8])
#         if group_number == 5:
#             IMMU_treat = np.array([0, 4, 8])
#         elif group_number == 7:
#             IMMU_treat = np.array([1, 5, 9])
#     elif group_number == 6:
#         TRAS_treat = np.array([0, 4, 8, 12, 16])
#         TUCA_treat = np.array([0, 4, 8, 12, 16])
#         IMMU_treat = np.array([0, 4, 8, 12, 16])
#     return TRAS_treat, TUCA_treat, IMMU_treat

# @jit(nopython=True)
# def model_exp_uncoupled(y, t, growth_rate, a_tras, b_tras, tau_tras, a_tuca, b_tuca, tau_tuca, a_immu, b_immu, tau_immu):
#     """
#     Compute the tumor volume with exponential growth and treatment effects.

#     Parameters:
#     - y: Initial tumor volume
#     - t: Current time
#     - growth_rate: Tumor growth rate
#     - a_tras, b_tras, tau_tras: Parameters for trastuzumab treatment
#     - a_tuca, b_tuca, tau_tuca: Parameters for tucatinib treatment
#     - a_immu, b_immu, tau_immu: Parameters for immunotherapy treatment

#     Returns:
#     - Tumor volume considering the treatment effects
#     """
#     treatment = 0

#     # Calculate treatment effect using vectorized operations
#     treatment_tras = np.sum(a_tras * np.exp(-b_tras * (t - tau_tras[tau_tras < t])))
#     treatment_tuca = np.sum(a_tuca * np.exp(-b_tuca * (t - tau_tuca[tau_tuca < t])))
#     treatment_immu = np.sum(a_immu * np.exp(-b_immu * (t - tau_immu[tau_immu < t])))

#     # Total treatment effect
#     treatment = treatment_tras + treatment_tuca + treatment_immu

#     # Calculate tumor volume
#     tumor = y
#     tumorVolume = growth_rate * tumor - treatment * tumor  # Assuming treatment reduces tumor volume

#     return tumorVolume

# @jit(nopython=True)
# def model_Vexp_uncoupled(y, t, growth_rate, a_tras, b_tras, tau_tras, a_tuca, b_tuca, tau_tuca, a_immu, b_immu, tau_immu, v_growth_rate, v_death_rate, c_tras):
#     """
#     Compute the tumor volume with exponential growth and treatment effects, including the well-vascularized fraction.

#     Parameters:
#     - y: Initial conditions [tumor volume, well-vascularized fraction]
#     - t: Current time
#     - growth_rate: Tumor growth rate
#     - a_tras, b_tras, tau_tras: Parameters for trastuzumab treatment (death rate, decay rate, administration times)
#     - a_tuca, b_tuca, tau_tuca: Parameters for tucatinib treatment (death rate, decay rate, administration times)
#     - a_immu, b_immu, tau_immu: Parameters for immunotherapy treatment (death rate, decay rate, administration times)
#     - v_growth_rate: Growth rate of the well-vascularized fraction
#     - v_death_rate: Death rate of the well-vascularized fraction due to tumor growth
#     - c_tras: Increase rate of well-vascularized fraction due to trastuzumab

#     Returns:
#     - (tumorVolume, vascularFraction): Tuple of tumor volume and well-vascularized fraction changes
#     """
#     tumor, vascular = y  # Unpack initial conditions

#     # Calculate treatment effects using vectorized operations
#     treatment_tras = np.sum(np.exp(-b_tras * (t - tau_tras[tau_tras < t])))
#     treatment_tuca = np.sum(np.exp(-b_tuca * (t - tau_tuca[tau_tuca < t])))
#     treatment_immu = np.sum(np.exp(-b_immu * (t - tau_immu[tau_immu < t])))

#     # Total treatment effect
#     treatment = a_tras * treatment_tras + a_tuca * treatment_tuca + a_immu * treatment_immu

#     # Calculate changes in tumor volume and well-vascularized fraction
#     tumorVolume = growth_rate * tumor - treatment * tumor * vascular
#     vascularFraction = (v_growth_rate + c_tras * treatment_tras) * (1 - vascular) - v_death_rate * tumor * vascular

#     return tumorVolume, vascularFraction

# @jit(nopython=True)
# def model_exp(y, t, growth_rate):
#     tumor = y
#     tumorVolume = growth_rate * tumor
#     return tumorVolume

# @jit(nopython=True)
# def model_Vexp(y, t, growth_rate, death_rate, v_growth_rate, v_death_rate):
#     tumor, vascular = y
#     tumorVolume = growth_rate * tumor - death_rate * tumor * vascular
#     vascularFraction = v_growth_rate * (1 - vascular) - v_death_rate * tumor * vascular
#     return tumorVolume,vascularFraction


    
# def get_vascular_info(model_extension):
#     isVascular = "V" in model_extension
#     scale = 2 if isVascular else 1
#     return isVascular, scale


# Define the logarithm of the prior probability for the model parameters
# def log_prior(theta, l_bound, u_bound):
#     """
#     Logarithm of the prior probability for the model parameters.

#     Parameters:
#     - theta: Model parameters
#     - l_bound: Lower bounds for the parameters
#     - u_bound: Upper bounds for the parameters

#     Returns:
#     - Logarithm of the prior probability
#     """
#     for l, p, u in zip(l_bound, theta, u_bound):
#         if not (l < p < u):
#             return -np.inf  # Return negative infinity for invalid parameter values
#     return 0.0

# def plot_data(all_data, nCols = 3, show = True, save = True, fontsize = '14', figure_name = './Figures/new_all_data'):
#     # Configure global settings for plot appearance
#     plt.rcParams['font.size'] = fontsize  # Set font size for all text in plots
    
#     # Check if either show or save is enabled
#     if not show and not save:
#         return

#     # Create a formatter for the y-axis that uses scientific notation when appropriate
#     formatter = mticker.ScalarFormatter(useMathText=True)
#     formatter.set_powerlimits((-3,2))  # Use scientific notation for numbers outside 10^-3 to 10^2
       
#     # Initialize variable to find the maximum value across all data for uniform y-axis scaling
#     all_max = 0
#     # Loop through each group of data
#     for group in all_data:
#         # Loop through each mouse in the group
#         for mouse in all_data[group]:
#             # Loop through each data set in the mouse's data (e.g., tumor volume, hypoxic fraction)
#             for j, data in enumerate(all_data[group][mouse]):
#                 if j == 0:  # Only check the first data type for maximum (assuming it's tumor volume)
#                     m_max = max(all_data[group][mouse][data][:,1])  # Find max value in this dataset
#                     if m_max > all_max:
#                         all_max = m_max  # Update overall max if this dataset's max is larger
    
#     # Round up the maximum y-value found to the nearest 1000 for nicer y-axis limits
#     rounded_max = math.ceil(all_max / 1000) * 1000
    
#     # Loop through each group again to create plots
#     for group in all_data:
#         nScenarios = len(all_data[group])  # Number of plots needed (one per mouse)
#         nRows = int(np.ceil(nScenarios / nCols))  # Calculate number of rows needed in the subplot grid
#         # Create subplots with calculated number of rows and columns, and set size
#         fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*12, nRows*5))
#         axes = axes.ravel()  # Flatten the axes array for easier indexing
    
#         # Loop over each mouse and its associated data
#         for i, mouse in enumerate(all_data[group]):
#             ax = axes[i]  # Get the current axis for the mouse
#             # Label the plot with the mouse identifier, replacing 'M' with 'Mouse'
#             ax.text(0.75, .9, '{:}) {:}'.format(string.ascii_uppercase[i], mouse.replace('M', 'Mouse ')),
#                     horizontalalignment='left', transform=ax.transAxes, fontsize=18)
#             ax.set_xlabel('Time (days)')  # Set the x-axis label
    
#             # Add vertical dashed lines at treatment days
#             if 'control' not in group:
#                 if '1' in group:
#                     ax.axvline(0,color='blue', linestyle='dashed',linewidth=2,label='IMT')
#                     ax.axvline(3,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(6,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(12,color='blue', linestyle='dashed',linewidth=2)
#                 elif '2' in group:
#                     ax.axvline(0,color='red', linestyle='dashed',linewidth=2,label='TRAS')
#                     ax.axvline(3,color='red', linestyle='dashed',linewidth=2)
#                     ax.axvline(6,color='red', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='red', linestyle='dashed',linewidth=2)
#                     ax.axvline(12,color='red', linestyle='dashed',linewidth=2)
#                 elif '3' in group:
#                     ax.axvline(0,color='purple', linestyle='dashed',linewidth=2,label='TRAS+IMT')
#                     ax.axvline(3,color='purple', linestyle='dashed',linewidth=2)
#                     ax.axvline(6,color='purple', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='purple', linestyle='dashed',linewidth=2)
#                     ax.axvline(12,color='purple', linestyle='dashed',linewidth=2)
#                 elif '7' in group:
#                     ax.axvline(1,color='blue', linestyle='dashed',linewidth=2,label='IMT')
#                     ax.axvline(5,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(0,color='green', linestyle='dashed',linewidth=2,label='T2')
#                     ax.axvline(4,color='green', linestyle='dashed',linewidth=2)
#                     ax.axvline(8,color='green', linestyle='dashed',linewidth=2)
#                 elif '4' in group:
#                     ax.axvline(0,color='green', linestyle='dashed',linewidth=2,label='T2')
#                     ax.axvline(4,color='green', linestyle='dashed',linewidth=2)
#                     ax.axvline(8,color='green', linestyle='dashed',linewidth=2)
#                 elif '5' in group or '6' in group:
#                     ax.axvline(0,color='turquoise', linestyle='dashed',linewidth=2,label='T2+IMT')
#                     ax.axvline(4,color='turquoise', linestyle='dashed',linewidth=2)
#                     ax.axvline(8,color='turquoise', linestyle='dashed',linewidth=2)
#                     if '6' in group:
#                         ax.axvline(12,color='turquoise', linestyle='dashed',linewidth=2)
#                         ax.axvline(16,color='turquoise', linestyle='dashed',linewidth=2)
#                 # Add legend to the first subplot
#                 if i == 0:
#                     ax.legend(loc=(0.75, 0.7), fancybox=False, shadow=False, frameon=False)

    
#             # Plot each type of data for the mouse
#             for j, data in enumerate(all_data[group][mouse]):
#                 if j == 0:  # First data type (primary y-axis)
#                     ax.set_ylabel(data)
#                     ax.plot(all_data[group][mouse][data][:,0], all_data[group][mouse][data][:,1],
#                             'o', color='black', linestyle='dashed', linewidth=2, markersize=10)
#                     ax.yaxis.set_major_formatter(formatter)
#                     ax.set_ylim((0, rounded_max))
#                     ax.set_xlim((-1, 26))
#                 else:  # Additional data types (secondary y-axis)
#                     ax2 = ax.twinx()
#                     ax2.set_ylabel(data, color='red')
#                     ax2.plot(all_data[group][mouse][data][:,0], all_data[group][mouse][data][:,1],
#                              'o', color='red', linestyle='dashed', linewidth=2, markersize=10)
#                     ax2.yaxis.set_major_formatter(formatter)
#                     ax2.spines['right'].set_color('red')
#                     ax2.tick_params(colors='red', which='both')
#                     ax2.set_ylim((0,1))  # Assuming secondary data ranges from 0 to 1
    
#         # Remove unused axes if there are any
#         for i in range(nScenarios, nRows * nCols):
#             fig.delaxes(axes[i])
    
#         # Save the figure to a PDF file, naming it after the group with spaces removed and converted to lowercase
#         if save:
#             plt.savefig(figure_name + '_' + group.lower().replace(" ", "") + '.pdf', bbox_inches = 'tight', pad_inches = 0.02)
#         if show:
#             plt.show()
#         else:
#             plt.close()
#     # Calculate number of scenarios from the length of all_data
#     nScenarios = len(all_data)
#     # Determine number of rows needed in the subplot grid
#     nRows = int(np.ceil(nScenarios / nCols))
    
#     # Create subplots with calculated number of rows and columns, and set figure size
#     fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*12, nRows*5))
#     # Flatten the axes array for easier indexing if in multiple rows
#     axes = axes.ravel()
    
#     # Iterate through each group in all_data
#     for i, group in enumerate(all_data):
#         ax = axes[i]  # Access the subplot for current group
#         # Add a text label to the subplot, formatting the group's index as a letter
#         ax.text(0.75, .9, '{:}) {:}'.format(string.ascii_uppercase[i], group),
#                 horizontalalignment='left', transform=ax.transAxes, fontsize=18)
#         ax.set_xlabel('Time (days)')  # Set the x-axis label for the subplot
        
#         # Loop over each mouse and its associated data
#         for mouse in all_data[group]:
    
#             # Add vertical dashed lines at treatment days
#             if 'control' not in group:
#                 if '1' in group:
#                     ax.axvline(0,color='blue', linestyle='dashed',linewidth=2,label='IMT')
#                     ax.axvline(3,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(6,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(12,color='blue', linestyle='dashed',linewidth=2)
#                 elif '2' in group:
#                     ax.axvline(0,color='red', linestyle='dashed',linewidth=2,label='TRAS')
#                     ax.axvline(3,color='red', linestyle='dashed',linewidth=2)
#                     ax.axvline(6,color='red', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='red', linestyle='dashed',linewidth=2)
#                     ax.axvline(12,color='red', linestyle='dashed',linewidth=2)
#                 elif '3' in group:
#                     ax.axvline(0,color='purple', linestyle='dashed',linewidth=2,label='TRAS+IMT')
#                     ax.axvline(3,color='purple', linestyle='dashed',linewidth=2)
#                     ax.axvline(6,color='purple', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='purple', linestyle='dashed',linewidth=2)
#                     ax.axvline(12,color='purple', linestyle='dashed',linewidth=2)
#                 elif '7' in group:
#                     ax.axvline(1,color='blue', linestyle='dashed',linewidth=2,label='IMT')
#                     ax.axvline(5,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(9,color='blue', linestyle='dashed',linewidth=2)
#                     ax.axvline(0,color='green', linestyle='dashed',linewidth=2,label='T2')
#                     ax.axvline(4,color='green', linestyle='dashed',linewidth=2)
#                     ax.axvline(8,color='green', linestyle='dashed',linewidth=2)
#                 elif '4' in group:
#                     ax.axvline(0,color='green', linestyle='dashed',linewidth=2,label='T2')
#                     ax.axvline(4,color='green', linestyle='dashed',linewidth=2)
#                     ax.axvline(8,color='green', linestyle='dashed',linewidth=2)
#                 elif '5' in group or '6' in group:
#                     ax.axvline(0,color='turquoise', linestyle='dashed',linewidth=2,label='T2+IMT')
#                     ax.axvline(4,color='turquoise', linestyle='dashed',linewidth=2)
#                     ax.axvline(8,color='turquoise', linestyle='dashed',linewidth=2)
#                     if '6' in group:
#                         ax.axvline(12,color='turquoise', linestyle='dashed',linewidth=2)
#                         ax.axvline(16,color='turquoise', linestyle='dashed',linewidth=2)
#                 #ax.legend(loc=(0.75, 0.7), fancybox=False, shadow=False, frameon=False)
        
#             # Plot data for each mouse in the group
#             for j, data in enumerate(all_data[group][mouse]):
#                 if j == 0:  # Assuming the first dataset is the primary one to plot
#                     ax.set_ylabel(data)  # Set the y-axis label with data type
#                     # Plot the data points, using different styles for visibility
#                     ax.plot(all_data[group][mouse][data][:,0], all_data[group][mouse][data][:,1], 
#                             'o', linestyle='dashed', linewidth=2, markersize=10)
#                     # Format y-axis using scientific notation if necessary
#                     ax.yaxis.set_major_formatter(formatter)
#                     # Set y-axis limits to include 0 to maximum rounded value calculated earlier
#                     ax.set_ylim((0, rounded_max))
#                     ax.set_xlim((-1, 26))
    
#     # Remove unused axes if the number of groups is less than the number of subplots
#     for i in range(nScenarios, nRows * nCols):
#         fig.delaxes(axes[i])
    
#     # Save the figure as a PDF, ensuring layout fits well without extra white space
#         if save:
#             plt.savefig(figure_name + '.pdf', bbox_inches = 'tight', pad_inches = 0.02)
#         if show:
#             plt.show()
#         else:
#             plt.close()
#     return



# def read_data(files_location, fmiso_qoi = 'hypoxicFraction', verbose = False):
#     # Use glob to find all files that match the given file location pattern
#     all_scenarios = sorted(glob.glob(files_location))

#     # Initialize a dictionary to store all data categorized by group and mouse
#     all_data = {}

#     # Iterate through each file path in the sorted list of file paths
#     for scn in all_scenarios:
#         # Extract mouse identifier from filename (assumes format: prefix_group_mouse_suffix.ext)
#         mouse = scn.split('_')[-2]
#         # Extract group name from path
#         group = scn.split('_')[-3].split('/')[-1]
        
#         # Ensure a dictionary for the group exists in all_data; if not, create it
#         if group not in all_data:
#             all_data[group] = {}
        
#         # Load hypoxic fraction data from a file corresponding to the current tumor volume file
#         m_data_hf = np.loadtxt(scn.replace('tumorVolume', fmiso_qoi))
#         # Load tumor volume data from the current file
#         m_data = np.loadtxt(scn)

#         # Store both data sets in a dictionary under the appropriate mouse key within the group
#         mouse_data = {'Tumor Volume (mm$^3$)': m_data, 'Well-Vascularized Fraction': m_data_hf}
#         all_data[group][mouse] = mouse_data

#     # Optionally print detailed information about each scenario being processed
#     if verbose:
#         for group in all_data:
#             to_print = ""
#             to_print += 'Scenario: ' + group
#             for mouse in all_data[group]:
#                 to_print += ', ' + mouse
#             print(to_print)

#     # Return the dictionary containing all the data
#     return all_data





















# def plot_maxll_solution(files_location, all_data, nCols = 3, show = True, save = True, fontsize = '14',figure_name = 'maxll_figure', bool_multiple_mice = False):
#     # Check if either show or save is enabled
#     if not show and not save:
#         return
    
#     all_chains = sorted(glob.glob(files_location))

#     # Configure global settings for plot appearance
#     plt.rcParams['font.size'] = fontsize  # Set font size for all text in plots
    
#     # Create a formatter for the y-axis that uses scientific notation when appropriate
#     formatter = mticker.ScalarFormatter(useMathText=True)
#     formatter.set_powerlimits((-3,2))  # Use scientific notation for numbers outside 10^-3 to 10^2

#     if bool_multiple_mice:
#         for chain in all_chains:
#             group = chain.split('/')[-1].split('_')[3]
#             model_extension = '_' + chain.split('_')[-1].split('.')[0]
#             if "V" in model_extension:
#                 isVascular = True
#                 scale = 2
#             else:
#                 isVascular = False
#                 scale = 1
#             nScenarios = len(all_data[group])
#             nRows = int(np.ceil(nScenarios / nCols))  # Calculate number of rows needed in the subplot grid
#             # Create subplots with calculated number of rows and columns, and set size
#             fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*12, nRows*5))
#             axes = axes.ravel()  # Flatten the axes array for easier indexing
#             ndim = len(all_data[group])
#             full_exp = []
#             full_model = []
#             for idx,mouse in enumerate(all_data[group]):
#                 ax = axes[idx]  # Get the current axis for the mouse
#                 key = next((k for k in full_data[group][mouse] if 'Tumor' in k), None)
#                 key_v = next((k for k in full_data[group][mouse] if 'Well' in k), None)
#                 ax.set_ylabel(key.replace('Tumor ',''))
#                 # Label the plot with the mouse identifier, replacing 'M' with 'Mouse'
#                 ax.text(0.05, .9, '{:}) {:}'.format(string.ascii_uppercase[idx], mouse.replace('M', 'Mouse ')),
#                         horizontalalignment='left', transform=ax.transAxes, fontsize=18)
#                 ax.plot(all_data[group][mouse][key][:,0], all_data[group][mouse][key][:,1],'o', color='black', markersize=10,label = 'Tumor')
#                 ax.set_xlabel('Time (days)')  # Set the x-axis label
                
#                 ax.yaxis.set_major_formatter(formatter)
#                 ax.set_xlim((-1, 26))
#                 # Extract relevant information from the sampler
#                 npzfile = np.load(chain)
#                 theta = npzfile['pars']

#                 # Extract the time array from the data for the current scenario
#                 time_array = full_data[group][mouse][key][:,0]
#                 time_array_v = full_data[group][mouse][key_v][:,0]
#                 # Create a boolean mask where elements of 'b' are in 'a'
#                 mask = np.isin(time_array, time_array_v)
                
#                 # Get the indices of elements in 'b' that are part of 'a'
#                 indices = np.where(mask)[0]
#                 is_subset = np.all(np.isin(time_array_v, time_array))
#                 if not is_subset:
#                     print(f'No subset')
            
#                 # Extract the parameters relevant to the current scenario
#                 parameters = tuple(theta[:-(ndim+1)*scale])
                
#                 # Extract the initial condition for the current scenario
#                 if isVascular:
#                     initial_condition = [theta[-(ndim+1)*scale+idx],theta[-(ndim+2)+idx]]
#                 else:
#                     initial_condition = theta[-(ndim+1)+idx]
               
#                 # Solve the model for the current scenario
#                 solution = solve_model(model_extension, time_array, parameters, initial_condition,"smooth")
        
#                 ax.plot(solution[:,0], solution[:,1], color='black', linestyle='dashed', linewidth=2, label = 'Model')
#                 if isVascular:
#                     ax2 = ax.twinx()
#                     ax2.set_ylabel(key_v, color='red')
#                     ax2.yaxis.set_major_formatter(formatter)
#                     ax2.spines['right'].set_color('red')
#                     ax2.tick_params(colors='red', which='both')
#                     ax2.set_ylim((0,1))  # Assuming secondary data ranges from 0 to 1
#                     ax2.plot(all_data[group][mouse][key_v][:,0], all_data[group][mouse][key_v][:,1],'o', color='red', markersize=10,label = 'Well-Vascularized')
#                     ax.set_xlabel('Time (days)')  # Set the x-axis label
#                     ax2.plot(solution[:,0], solution[:,2], color='red', linestyle='dashed', linewidth=2, label = 'Model')

#                 ax.legend(loc=(0.05, 0.6), fancybox=False, shadow=False, frameon=False)
#                 solution = solve_model(model_extension, time_array, parameters, initial_condition)
#                 full_model.append(solution[:,0])
#                 full_exp.append(all_data[group][mouse][key][:,1])
#                 pccT = fp.pcc(solution[:,0], all_data[group][mouse][key][:,1])
#                 cccT = fp.ccc(solution[:,0], all_data[group][mouse][key][:,1])
#                 mapeT = fp.mape(solution[:,0], all_data[group][mouse][key][:,1])
#                 nrmseT = fp.nrmse(solution[:,0], all_data[group][mouse][key][:,1])
#                 ax.text(0.5,.05,'CCC/PCC/MAPE = {:.2f}/{:.2f}/{:.2f}%'.format(cccT,pccT,mapeT),horizontalalignment='left',transform=ax.transAxes)
            
#             final_exp = np.concatenate(full_exp, axis=0)  # Adjust axis if needed
#             final_model = np.concatenate(full_model, axis=0)  # Adjust axis if needed
#             range_max = np.max(np.concatenate([final_exp, final_model]))
#             rounded_max = math.ceil(range_max / 1000) * 1000
        
#             # Remove unused axes if there are any
#             for idx,mouse in enumerate(all_data[group]):
#                 ax = axes[idx]  # Get the current axis for the mouse
#                 ax.set_ylim((0,rounded_max))
#             for i in range(nScenarios, nRows * nCols):
#                 fig.delaxes(axes[i])
            
#             # Save the figure to a PDF file, naming it after the group with spaces removed and converted to lowercase
#             if save:
#                 plt.savefig(figure_name + '.pdf', bbox_inches = 'tight', pad_inches = 0.02)
#             if show:
#                 plt.show()
#             else:
#                 plt.close()
#             fig, ax = plt.subplots(figsize=(5, 5))
#             pccT = fp.pcc(final_model, final_exp)
#             cccT = fp.ccc(final_model, final_exp)
#             mapeT = fp.mape(final_model, final_exp)
#             ax.text(0.025,.85,'CCC/PCC = {:.2f}/{:.2f}\nMAPE = {:.2f}%'.format(cccT,pccT,mapeT),horizontalalignment='left',transform=ax.transAxes)
#             ax.set_xlabel('Data - Tumor volume (mm³)')
#             ax.set_ylabel('Model - Tumor volume (mm³)')
#             ax.plot(final_exp,final_model, 'o',color='blue')
#             line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
#             transform = ax.transAxes
#             line.set_transform(transform)
#             ax.add_line(line)
#             ax.set_ylim((0,rounded_max))
#             ax.set_xlim((0,rounded_max))
#             ax.yaxis.set_major_formatter(formatter)
#             ax.xaxis.set_major_formatter(formatter)
#             # Save the figure to a PDF file, naming it after the group with spaces removed and converted to lowercase
#             if save:
#                 plt.savefig(figure_name + '_sp.pdf', bbox_inches = 'tight', pad_inches = 0.02)
#             if show:
#                 plt.show()
#             else:
#                 plt.close()
#     else:
#         nScenarios = len(all_chains)
#         nRows = int(np.ceil(nScenarios / nCols))  # Calculate number of rows needed in the subplot grid
#         # Create subplots with calculated number of rows and columns, and set size
#         fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*12, nRows*5))
#         axes = axes.ravel()  # Flatten the axes array for easier indexing
    
#         full_exp = []
#         full_model = []
    
#         for idx,chain in enumerate(all_chains):
#             ax = axes[idx]  # Get the current axis for the mouse
#             model_extension = '_' + chain.split('_')[-1].split('.')[0]
#             if "V" in model_extension:
#                 isVascular = True
#                 scale = 2
#             else:
#                 isVascular = False
#                 scale = 1
#             mouse = chain.split('_')[-2]
#             group = chain.split('_')[-3]
#             key = next((k for k in full_data[group][mouse] if 'Tumor' in k), None)
#             ax.set_ylabel(key)
#             # Label the plot with the mouse identifier, replacing 'M' with 'Mouse'
#             ax.text(0.05, .9, '{:}) {:}'.format(string.ascii_uppercase[idx], mouse.replace('M', 'Mouse ')),
#                     horizontalalignment='left', transform=ax.transAxes, fontsize=18)
#             ax.plot(all_data[group][mouse][key][:,0], all_data[group][mouse][key][:,1],'o', color='black', markersize=10,label = 'Tumor')
#             key_v = next((k for k in full_data[group][mouse] if 'Well' in k), None)
#             ax2 = ax.twinx()
#             ax2.set_ylabel(key_v, color='red')
#             ax2.yaxis.set_major_formatter(formatter)
#             ax2.spines['right'].set_color('red')
#             ax2.tick_params(colors='red', which='both')
#             ax2.set_ylim((0,1))  # Assuming secondary data ranges from 0 to 1
#             ax2.plot(all_data[group][mouse][key_v][:,0], all_data[group][mouse][key_v][:,1],'o', color='red', markersize=10,label = 'Well-Vascularized')
#             ax.set_xlabel('Time (days)')  # Set the x-axis label
            
#             ax.yaxis.set_major_formatter(formatter)
#             ax.set_xlim((-1, 26))
#             # Extract relevant information from the sampler
#             npzfile = np.load(chain)
#             theta = npzfile['pars']
        
#             # Extract the time array from the data for the current scenario
#             time_array = all_data[group][mouse][key][:,0]
#             time_array_v = all_data[group][mouse][key_v][:,0]
#             # Create a boolean mask where elements of 'b' are in 'a'
#             mask = np.isin(time_array, time_array_v)
            
#             # Get the indices of elements in 'b' that are part of 'a'
#             indices = np.where(mask)[0]
#             is_subset = np.all(np.isin(time_array_v, time_array))
#             if not is_subset:
#                 print(f'No subset')
    
#             # Extract the parameters relevant to the current scenario
#             parameters = tuple(theta[:-2*scale])
            
#             # Extract the initial condition for the current scenario
#             if isVascular:
#                 initial_condition = [theta[-4],theta[-3]]
#             else:
#                 initial_condition = theta[-2]
           
#             # Solve the model for the current scenario
#             solution = solve_model(model_extension, time_array, parameters, initial_condition,"smooth")
    
#             ax.plot(solution[:,0], solution[:,1], color='black', linestyle='dashed', linewidth=2, label = 'Model')
#             if isVascular:
#                 ax2.plot(solution[:,0], solution[:,2], color='red', linestyle='dashed', linewidth=2, label = 'Model')
            
#             ax.legend(loc=(0.05, 0.6), fancybox=False, shadow=False, frameon=False)
#             solution = solve_model(model_extension, time_array, parameters, initial_condition)
#             full_model.append(solution[:,0])
#             full_exp.append(all_data[group][mouse][key][:,1])
#             pccT = fp.pcc(solution[:,0], all_data[group][mouse][key][:,1])
#             cccT = fp.ccc(solution[:,0], all_data[group][mouse][key][:,1])
#             mapeT = fp.mape(solution[:,0], all_data[group][mouse][key][:,1])
#             nrmseT = fp.nrmse(solution[:,0], all_data[group][mouse][key][:,1])
#             ax.text(0.5,.05,'CCC/PCC/MAPE = {:.2f}/{:.2f}/{:.2f}%'.format(cccT,pccT,mapeT),horizontalalignment='left',transform=ax.transAxes)
    
#         final_exp = np.concatenate(full_exp, axis=0)  # Adjust axis if needed
#         final_model = np.concatenate(full_model, axis=0)  # Adjust axis if needed
#         range_max = np.max(np.concatenate([final_exp, final_model]))
#         rounded_max = math.ceil(range_max / 1000) * 1000
    
#         # Remove unused axes if there are any
#         for idx,chain in enumerate(all_chains):
#             ax = axes[idx]  # Get the current axis for the mouse
#             ax.set_ylim((0,rounded_max))
#         for i in range(nScenarios, nRows * nCols):
#             fig.delaxes(axes[i])
        
#         # Save the figure to a PDF file, naming it after the group with spaces removed and converted to lowercase
#         if save:
#             plt.savefig(figure_name + '.pdf', bbox_inches = 'tight', pad_inches = 0.02)
#         if show:
#             plt.show()
#         else:
#             plt.close()
#         fig, ax = plt.subplots(figsize=(5, 5))
#         pccT = fp.pcc(final_model, final_exp)
#         cccT = fp.ccc(final_model, final_exp)
#         mapeT = fp.mape(final_model, final_exp)
#         ax.text(0.025,.85,'CCC/PCC = {:.2f}/{:.2f}\nMAPE = {:.2f}%'.format(cccT,pccT,mapeT),horizontalalignment='left',transform=ax.transAxes)
#         ax.set_xlabel('Data - Tumor volume (mm³)')
#         ax.set_ylabel('Model - Tumor volume (mm³)')
#         ax.plot(final_exp,final_model, 'o',color='blue')
#         line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
#         transform = ax.transAxes
#         line.set_transform(transform)
#         ax.add_line(line)
#         ax.set_ylim((0,rounded_max))
#         ax.set_xlim((0,rounded_max))
#         ax.yaxis.set_major_formatter(formatter)
#         ax.xaxis.set_major_formatter(formatter)
#         # Save the figure to a PDF file, naming it after the group with spaces removed and converted to lowercase
#         if save:
#             plt.savefig(figure_name + '_sp.pdf', bbox_inches = 'tight', pad_inches = 0.02)
#         if show:
#             plt.show()
#         else:
#             plt.close()
#     return