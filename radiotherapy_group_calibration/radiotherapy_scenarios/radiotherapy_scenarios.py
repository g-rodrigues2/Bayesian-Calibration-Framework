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
import pdse_project as pdse
import seaborn as sns
import scipy.stats as stats
import sys
# import arviz as az


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
    sxy = np.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
    rhoc = 2 * sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

# Pearson Correlation Coefficient
def pcc(x, y):
    """
    Calculate the Pearson Correlation Coefficient (PCC).

    Parameters:
    - x, y: Input data arrays

    Returns:
    - PCC value
    """
    sxy = np.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
    rho = sxy / (np.std(x) * np.std(y))
    return rho

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
    actual, pred = np.array(actual), np.array(pred)
    rmse = np.sqrt(np.sum((actual - pred)**2) / len(actual))
    return rmse / np.mean(actual) * 100

# Mean Absolute Percentage Error
def mape(actual, pred): 
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters:
    - actual: Actual values
    - pred: Predicted values

    Returns:
    - MAPE value
    """
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

#-------------------------- Modelos de Radioterapia ---------------------------------------

@jit(nopython=True)
def model_DamageRepairTimeDelay(y, t, growth_rate, carrying_capacity, a_radiation, b_radiation, delay, tau_radiation, dose_schedule):
    """
    Modelo logístico com efeito da radioterapia (LQ model).
    """
    # Crescimento logístico
    tumor = y
    tumor_volume = growth_rate * tumor * (1 - tumor / carrying_capacity)

    # Efeito do tratamento (radiação)
    treatment_effect = 0
    for dose, tau in zip(dose_schedule, tau_radiation):
        if t >= tau + delay:  # Aplica o efeito de delay
            treatment_effect += a_radiation * dose * np.exp(-b_radiation * (t - tau - delay))

    # Ajuste do volume tumoral pelo tratamento
    tumor_volume -= treatment_effect * tumor  # Redução do tumor pela radioterapia

    return tumor_volume

def solve_model(model_extension, time_array, parameters, initial_condition, treatment_days, dose, type_sol='data'):
    model_name = 'model' + model_extension
    model_func = globals()[model_name]  # Obtém a função do modelo (ex: model_LQ)

    # Definir o número de doses e dias de tratamento a partir dos dados fornecidos
    tau_radiation = treatment_days
    dose_schedule = [dose] * len(tau_radiation)  # Todas as doses são de 2Gy

    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p - bgn_p) / 0.1) + 1)
        sol = odeint(model_func, t=time_array, y0=[initial_condition], args=(*parameters, tau_radiation, dose_schedule), mxstep=2000)
        return np.column_stack((time_array, sol))
    else:
        return odeint(model_func, t=time_array, y0=[initial_condition], args=(*parameters, tau_radiation, dose_schedule), mxstep=2000)

def log_likelihood(theta, model_extension, full_data, group):
    
    ll = 0
    
    rs = 8.89429860e-02  # Crescimento para radiation_sensitive
    rr = 1.15631515e-01  # Crescimento para radiation_resistant
    carrying_capacity = 3.69671552e+03  # Capacidade de suporte compartilhada

    variance = theta[-1] ** 2

    a_s = theta[0]
    a_r = theta[1]  
    b_s = theta[2]    
    b_r= theta[3]
    d_s = theta[4]
    d_r = theta[5]

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

            treatment_params = (a_radiation, b_radiation, delay)

            initial_condition = theta[-(ndim+1)+idx]

            solution = solve_model(model_extension, time_array, (growth_rate, carrying_capacity, *treatment_params), 
                               initial_condition, treatment_days, dose)
            # print('teste\n', solution)
            observed_volume = mouse_data['data'][:, 1]
            ll += -0.5 * np.sum((solution[:, 0] - observed_volume) ** 2 / variance + np.log(2 * np.pi) + np.log(variance))

            idx += 1

    return ll

# Define the logarithm of the prior probability for the model parameters
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
            return -np.inf  # Return negative infinity for invalid parameter values
    return 0.0

def log_probability(theta, l_bound, u_bound, model_extension, full_data, group):
    """
    Calcula a probabilidade total (verossimilhança + prior) para o grupo inteiro.
    
    Parameters:
    - theta: Parâmetros do modelo.
    - l_bound: Limites inferiores dos parâmetros.
    - u_bound: Limites superiores dos parâmetros.
    - model_extension: O modelo a ser usado (por exemplo, '_exp', '_log').
    - full_data: Dicionário contendo os dados dos grupos e ratos.
    - group: O grupo de tratamento (por exemplo, 'control_sensitive' ou 'control_resistant').
    
    Returns:
    - Log da probabilidade total.
    """
    # Calcular o log do prior (probabilidade dos parâmetros)
    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf  # Retorna -infinito se os parâmetros estiverem fora dos limites

    # Somar o log da verossimilhança para todos os ratos do grupo
    return lp + log_likelihood(theta, model_extension, full_data, group)
    

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


def plot_maxll_solution(files_location_sensitive, files_location_resistant, full_data, nCols=3, show=True, save=True, fontsize='14', figure_name='maxll_figure'):
    if not show and not save:
        return 
    
    formatter = pdse.configure_plot_settings(fontsize)
    plt.rcParams['font.size'] = fontsize
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))

    # Verificar se os arquivos foram passados corretamente
    theta_sensitive = None
    theta_resistant = None

    if files_location_sensitive is not None:
        npzfile_sensitive = np.load(files_location_sensitive)
        theta_sensitive = npzfile_sensitive['pars']
        print("Parâmetros carregados para radiation_sensitive:\n", theta_sensitive)

    if files_location_resistant is not None:
        npzfile_resistant = np.load(files_location_resistant)
        theta_resistant = npzfile_resistant['pars']
        print("Parâmetros carregados para radiation_resistant:\n", theta_resistant)

    max_times = find_max_time_per_group(full_data)

    # Parâmetros fixos que não estão no vetor `theta`
    rs = 8.89429860e-02  # Crescimento para radiation_sensitive
    rr = 1.15631515e-01  # Crescimento para radiation_resistant
    carrying_capacity = 3.69671552e+03  # Capacidade de suporte

        # Iterar sobre os grupos e plotar os melhores ajustes
    for group in ['radiation_sensitive', 'radiation_resistant']:
        if group == 'radiation_sensitive' and theta_sensitive is not None:
            print(f"Processando: {group}")
            theta = theta_sensitive
        elif group == 'radiation_resistant' and theta_resistant is not None:
            print(f"Processando: {group}")
            theta = theta_resistant
        else:
            continue  # Se o grupo não tem dados, pula para o próximo

        # Configurar os subplots
        nScenarios = len(full_data[group])
        nRows = int(np.ceil(nScenarios / nCols))
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols * 11, nRows * 5))
        axes = axes.ravel()

        full_exp = []
        full_model = [] 

        model_extension = "_DamageRepairTimeDelay" 

        ndim = len(full_data[group])
        treatment_params = (tuple(theta[:-(ndim+1)])) # Extrair `a_s`, `b_radiation`, e `delay` corretamente
        print(f'Params de tratamento = [{treatment_params}]\nQtd. de parametros = {len(theta)}\nQtd. de animais = {ndim}\n'  )  

        max_time_group = max_times[group]

        # Plotar cada rato no grupo
        for idx, mouse_data in enumerate(full_data[group]):
            mouse_name = mouse_data['name']
            ax = axes[idx]

            # Acessar os dados de volume tumoral diretamente
            data = mouse_data['data']
            treatment_days = mouse_data['treatment_days']
            dose = 2  # Dose fixa de 2Gy

            ax.set_ylabel('Tumor Volume (mm³)')
            ax.text(0.05, .9, f'{string.ascii_uppercase[idx]}) {mouse_name}', horizontalalignment='left', transform=ax.transAxes, fontsize=18)

            # Plotar os dados reais (exp)
            ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10, label='Tumor Volume')
            ax.set_xlabel('Time (days)')
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim((-1, max_time_group + 1))

            # Adicionar linhas verticais para os dias de tratamento
            for t_day in treatment_days:
                ax.axvline(x=t_day, color='green', linestyle='--', linewidth=1)

            # Determinar o parâmetro de taxa de crescimento com base no grupo
            if group == 'radiation_sensitive':
                growth_rate = rs
            elif group == 'radiation_resistant':
                growth_rate = rr

            initial_condition = theta[-(ndim + 1) + idx]  

            solution = solve_model(model_extension, data[:, 0], (growth_rate, carrying_capacity, *treatment_params), 
                               initial_condition, treatment_days, dose, 'smooth')
            

            # Filtrar as soluções para os tempos que correspondem a `data[:, 0]`
            mask = np.isin(solution[:, 0], data[:, 0])
            matched_solution_times = solution[mask, 0]
            matched_solution_volumes = solution[mask, 1]

            # Plotar a solução do modelo ajustado
            ax.plot(matched_solution_times, matched_solution_volumes, color='black', linestyle='dashed', linewidth=2, label='Model')

            # Adicionar a solução e os dados reais às listas para análise posterior
            full_model.append(matched_solution_volumes)
            full_exp.append(data[:, 1])

            # Calcular métricas de ajuste (CCC, PCC)
            pccT = pcc(matched_solution_volumes, data[:, 1])
            cccT = ccc(matched_solution_volumes, data[:, 1])

            # Mostrar as métricas no gráfico
            if group == 'radiation_sensitive':
                ax.text(0.5, .05, f'CCC/PCC = {cccT:.2f}/{pccT:.2f}', horizontalalignment='left', transform=ax.transAxes)
            else:
                mapeT = pdse.mape(matched_solution_volumes, data[:, 1])
                # Mostrar as métricas no gráfico
                ax.text(0.5, .05, 'CCC/PCC/MAPE = {:.2f}/{:.2f}/{:.2f}%'.format(cccT, pccT, mapeT), horizontalalignment='left', transform=ax.transAxes)
        
        # Ajustar o nome do arquivo para salvar os gráficos separadamente por grupo
        version = 'V1'
        group_figure_name = f'{figure_name}{version}'  # Nome para salvar o arquivo

        # Finalizar o gráfico para o grupo
        pdse.finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model, save, show, group_figure_name, formatter)
        pdse.color_plot_scatter(full_exp, full_model, save, show, group, group_figure_name, formatter)

    return


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

max_ic = find_max_initial_condition(full_data)
max_ic = max_ic[0]
# print(max_ic)


def define_bounds_labels(model_extension, bool_multiple_mice = False):
    labels = []
    l_bound = []
    u_bound = []
    ic_max = max_ic*2
    if model_extension == '_DamageRepairTimeDelay':
        labels = ["a_s", "a_r", "b_s", "b_r", "d_s", "d_r"]
        l_bound = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        u_bound = [0.1, 0.1, 0.1, 2.0, 10.0, 10.0]
  
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

# # Função principal
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

    version = 'V1'
    run_calibration = True
    if run_calibration:
        for model_extension in ['_DamageRepairTimeDelay']: #'_exp', '_mendel', '_bert', '_lin', '_gomp', '_surf', '_log']:

            original_group = ['radiation_sensitive', 'radiation_resistant']                        
            group_size = 0
            new_group = []
            for group in ['radiation_sensitive', 'radiation_resistant']:
            
                group_size += len(full_data[group])        
                new_group.append(group)
                    

            l_bound, u_bound, labels = define_bounds_labels(model_extension, bool_multiple_mice=True)
            group = new_group
            # print(group)
            # print(labels)
            
            # Obtenha o número de dimensões com base nos limites
            ndim = len(l_bound)
            
            # Define o número de walkers para o MCMC
            nwalkers = 2 * ndim
            
            # Nome do arquivo para o backend
            filename = f"./reinitiate_files/calibration{version}.h5"

            
            # Verificar se o arquivo existe
            if not os.path.exists(filename):
                print("Arquivo HDF5 não encontrado. Criando um novo arquivo...")
                backend = emcee.backends.HDFBackend(filename)
                backend.reset(nwalkers, ndim)  # Resetar backend para iniciar uma nova execução
                
                # Inicializar posições aleatórias para os walkers
                pos = np.zeros((nwalkers, ndim))
                for i in range(ndim):
                    pos[:, i] = np.random.uniform(low=l_bound[i], high=u_bound[i], size=nwalkers)
            else:
                print(f"Arquivo HDF5 encontrado: {filename}")
                backend = emcee.backends.HDFBackend(filename)
                print(f"Progresso atual no backend: {backend.iteration} iterações.")
                
                # Caso o arquivo exista, inicializar `pos` como `None` para continuar
                pos = None

            # Define o tamanho da cadeia de Markov adicional
            additional_chain_size = 10  # Número de passos adicionais    

            print(f'Calibrating {group}, using model {model_extension.split("_")[-1]}')
            
            # Calibração do modelo para o grupo usando MCMC - TACC
            # n_cores = cpu_count()
            # optimal_cores = int(n_cores * 0.75)
            # print(f"Número de núcleos usados: {optimal_cores}")
            with Pool(processes=cpu_count()) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability, 
                    args=(l_bound, u_bound, model_extension, full_data, group),
                    pool = pool,
                    backend=backend
                )
                # Se já existe progresso, continue; caso contrário, inicie nova cadeia
                if backend.iteration > 0:
                    sampler.run_mcmc(None, additional_chain_size, progress=True)  # Continue de onde parou
                else:
                    sampler.run_mcmc(pos, additional_chain_size, progress=True)  # Inicie uma nova execução
            
            flat_ll = sampler.get_log_prob(flat=True)
            flat_chain = sampler.get_chain(flat=True)
            max_pos = flat_ll.argmax(axis=0)
            best_pars = flat_chain[max_pos]
            total_size = ((additional_chain_size - additional_chain_size//4)*ndim*2)
            goal_size = 10000
            scale_corner = max(1, total_size // goal_size) # adcionei o max para garantir que scale_corner nunca será zero
            # size_discard = chain_size - 10000 // nwalkers
            save_chain = sampler.get_chain(discard=additional_chain_size // 4, flat=True, thin = scale_corner)
            
            # ---------------------------------------------------
            # '''
            # BLOCO PARA AVALIAR A CONVERGENCIA DAS CADEIAS DE MARKOV
            # UTILIZANDO O CRITÉRIO GELMAN-RUBIN
            # '''
            # # Obter as cadeias do sampler
            # chains = sampler.get_chain(discard=additional_chain_size // 4, flat=False)
            # # Transformar as cadeias em um Dataset
            # chains_dataset = az.convert_to_dataset(chains)
            # # Calcular o Rhat
            # rhat = az.rhat(chains_dataset)
            # print("Rhat values:", rhat)
            # for param in rhat.data_vars:  # Itera sobre as variáveis do dataset
            #     values = rhat[param].values  # Obtém os valores do Rhat para o parâmetro
            #     print(f"Parâmetro {param}:")
            #     for i, value in enumerate(values):
            #         print(f"  Subíndice {i}: Rhat = {value:.3f}")
            
            # -----------------------------------------------------------------------

            # print(len(save_chain))
            print(f'vetor theta para verificacao:\n {best_pars} \n')
            print(f"Total steps completed: {backend.iteration}")
            group = original_group 

            # Separar as informações com base no grupo
            for g in group:
                if g == 'radiation_sensitive':
                    # print('sensitive_chain')
                    num_ic_sensitive = len(full_data['radiation_sensitive'])  
                    labels_sensitive = ["a_s", "b_s", "d_s"] + [f"ic{i+1}" for i in range(num_ic_sensitive)] + ["std"]
                    print(labels_sensitive)
                    # Extrair parâmetros para control_sensitive
                    chain_sensitive = save_chain[:, [0, 2, 4] + list(range(6, 6 + num_ic_sensitive)) + [-1]]  # rs, cc, ic1 até ic9, std           
                    
                    # Encontrar as posições corretas em best_pars para sensitive
                    best_pars_sensitive = best_pars[[0, 2, 4] + list(range(6, 6 + num_ic_sensitive)) + [-1]]
                    # print(f'best_pars_sensitive: {best_pars_sensitive}')

                    # Salvar as informações da chain
                    np.savetxt(f'./Output_Calibration/multi_chain_radiation_sensitive{model_extension}{version}.gz', chain_sensitive)
                    np.savez(f'./Output_Calibration/multi_ll_pars_radiation_sensitive{model_extension}{version}.npz', max_ll=max(flat_ll), pars=best_pars_sensitive)
                    
                    # Plotar para o grupo control_sensitive
                    plot_maxll_solution(
                        f'./Output_Calibration/multi_ll_pars_radiation_sensitive{model_extension}{version}.npz',
                        None,  # Não precisamos passar um arquivo para resistant neste momento
                        full_data,
                        nCols=4,
                        show=False,
                        save=True,
                        figure_name=f'./Output_Calibration/multi_max_ll_radiation_sensitive{model_extension}'
                    )

                    # Gerar corner plot para control_sensitive
                    fig_sensitive = corner.corner(chain_sensitive, labels=labels_sensitive, truths = best_pars_sensitive)
                    plt.savefig(f'./Output_Calibration/multi_corner_radiation_sensitive{model_extension}{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                    plt.close()

                    # Gerar gráfico das chains para control_sensitive
                    fig, axes = plt.subplots(nrows=len(labels_sensitive), ncols=1, figsize=(10, len(labels_sensitive) * 2), sharex=True)
                    axes = axes.ravel()
                    # Pegar as cadeias completas
                    samples = sampler.get_chain()

                    indices_sensitive = [0, 2, 4] + list(range(6, 6 + num_ic_sensitive)) + [-1]  # Inclui `a_s`, `b_radiation`, `delay`, ICs, e `std`

                    # Obter as amostras corretas para 'radiation_sensitive'
                    samples_sensitive = samples[:, :, indices_sensitive]  # Seleciona apenas os parâmetros relevantes para o grupo

                    for i in range(len(labels_sensitive)):
                        ax = axes[i]
                        ax.plot(samples_sensitive[:, :, i], "k", alpha=0.3)  # Plotar apenas os parâmetros filtrados
                        ax.set_xlim(0, len(samples_sensitive))
                        ax.set_ylabel(labels_sensitive[i])
                        ax.yaxis.set_label_coords(-0.1, 0.5)

                    axes[-1].set_xlabel("step number")
                    plt.savefig(f'./Output_Calibration/multi_chain_radiation_sensitive{model_extension}{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                    plt.close()# Gerar a solução de máxima verossimilhança
                
                elif g == 'radiation_resistant':
                    # print('resistant chain')
                    # Extrair parâmetros para control_resistant
                    num_ic_resistant = len(full_data['radiation_resistant'])  # Número de camundongos em 'radiation_resistant'
                    labels_resistant = ["a_r", "b_r", "d_r"] + [f"ic{i+1}" for i in range(num_ic_resistant)] + ["std"]
                    print(labels_resistant)
                    chain_resistant = save_chain[:, [1, 3, 5] + list(range(6 + num_ic_sensitive, 6 + num_ic_sensitive + num_ic_resistant)) + [-1]]  # rr, cc, ic10 até ic14, std
                    # print('chain_resistant', chain_resistant)
                    # print('tamanho da chain:', len(chain_resistant[0])) 
                    # Encontrar as posições corretas em best_pars para resistant
                    best_pars_resistant = best_pars[[1, 3, 5] + list(range(6 + num_ic_sensitive, 6 + num_ic_sensitive + num_ic_resistant))  + [-1]]
                    # print(f'best_pars_resistant: {best_pars_resistant}')
                    
                    
                    # Salvar as informações da chain
                    np.savetxt(f'./Output_Calibration/multi_chain_radiation_resistant{model_extension}{version}.gz', chain_resistant)
                    np.savez(f'./Output_Calibration/multi_ll_pars_radiation_resistant{model_extension}{version}.npz', max_ll=max(flat_ll), pars=best_pars_resistant)

                    plot_maxll_solution(
                        None,  # Não precisamos passar um arquivo para sensitive neste momento
                        f'./Output_Calibration/multi_ll_pars_radiation_resistant{model_extension}{version}.npz',
                        full_data,
                        nCols=4,
                        show=False,
                        save=True,
                        figure_name=f'./Output_Calibration/multi_max_ll_radiation_resistant{model_extension}'
                    )

                    # Gerar corner plot para control_resistant
                    fig_resistant = corner.corner(chain_resistant, labels=labels_resistant, truths = best_pars_resistant)
                    plt.savefig(f'./Output_Calibration/multi_corner_radiation_resistant{model_extension}{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                    plt.close()

                    # Gerar gráfico das chains para control_resistant
                    fig, axes = plt.subplots(nrows=len(labels_resistant), ncols=1, figsize=(10, len(labels_resistant) * 2), sharex=True)
                    axes = axes.ravel()
                    samples = sampler.get_chain()

                    
                    indices_resistant = [1, 3, 5] + list(range(6 + num_ic_sensitive, 6 + num_ic_sensitive + num_ic_resistant))  + [-1]  # Inclui `a_s`, `b_radiation`, `delay`, ICs, e `std`

                    # Obter as amostras corretas para 'radiation_sensitive'
                    samples_resistant = samples[:, :, indices_resistant]  # Seleciona apenas os parâmetros relevantes para o grupo

                    # Ajuste os rótulos para os parâmetros `resistant`
                    labels_resistant = ["a_r", "b_r", "d_r"] + [f"ic{i+1}" for i in range(num_ic_resistant)] + ["std"]
                    # print("Labels para resistant:", labels_resistant)
                    # print('camundongos no grupo resistant', num_ic_resistant)   

                    for i in range(len(labels_resistant)):
                        ax = axes[i]
                        ax.plot(samples_resistant[:, :, i], "k", alpha=0.3)  # Plotar apenas os parâmetros filtrados
                        ax.set_xlim(0, len(samples_resistant))
                        ax.set_ylabel(labels_resistant[i])
                        ax.yaxis.set_label_coords(-0.1, 0.5)

                    plt.savefig(f'./Output_Calibration/multi_chain_radiation_resistant{model_extension}{version}.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                    plt.close()