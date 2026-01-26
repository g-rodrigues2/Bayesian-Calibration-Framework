import glob
import math
import string
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import scipy.stats as stats


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

def cor_test(x, y, method='pearson'):
    """
    Perform a correlation test analogous to R's cor.test().
    Supports 'pearson', 'spearman', and 'kendall'.

    Returns:
        dict(method, correlation, p_value, conf_int (tuple), n)
        * conf_int (95%) é calculado apenas para Pearson via Fisher z-transform.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must have the same length")

    if method == 'pearson':
        r, p_value = stats.pearsonr(x, y)
        n = len(x)
        # Fisher z-transform para IC 95%
        if n > 3 and abs(r) < 1:
            z = np.arctanh(r)
            se = 1.0 / np.sqrt(n - 3)
            delta = 1.96 * se
            ci_lower, ci_upper = np.tanh([z - delta, z + delta])
        else:
            ci_lower, ci_upper = np.nan, np.nan
    elif method == 'spearman':
        r, p_value = stats.spearmanr(x, y)
        ci_lower, ci_upper = np.nan, np.nan
        n = len(x)
    elif method == 'kendall':
        r, p_value = stats.kendalltau(x, y)
        ci_lower, ci_upper = np.nan, np.nan
        n = len(x)
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'.")

    return {
        'method': method,
        'correlation': r,
        'p_value': p_value,
        'conf_int': (ci_lower, ci_upper),
        'n': n
    }

def ccc_test(x, y, alpha=0.05):
    """
    Teste de significância para o Concordance Correlation Coefficient (Lin, 1989).
    Retorna CCC, estatística t aproximada, p-valor e IC95% (Fisher z).
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays must have the same length")
    n = x.shape[0]
    if n < 3:
        return {'ccc': np.nan, 't_stat': np.nan, 'p_value': np.nan,
                'conf_int': (np.nan, np.nan), 'n': n}

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    if var_x == 0 or var_y == 0:
        return {'ccc': np.nan, 't_stat': np.nan, 'p_value': np.nan,
                'conf_int': (np.nan, np.nan), 'n': n}

    cov_xy = np.cov(x, y, bias=True)[0, 1]
    ccc_value = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)

    # Estatística t aproximada (H0: rho_c = 0)
    if abs(ccc_value) < 1:
        t_stat = ccc_value * np.sqrt((n - 2) / (1 - ccc_value ** 2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    else:
        t_stat, p_value = np.nan, 0.0  # concordância perfeita -> p ~ 0

    # IC95% via Fisher z (válido para |rho|<1 e n>3)
    if (n > 3) and (abs(ccc_value) < 1):
        z = np.arctanh(ccc_value)
        se = 1.0 / np.sqrt(n - 3)
        z_crit = 1.96
        lo, hi = np.tanh([z - z_crit * se, z + z_crit * se])
    else:
        lo, hi = np.nan, np.nan

    return {
        'ccc': ccc_value,
        't_stat': t_stat,
        'p_value': p_value,
        'conf_int': (lo, hi),
        'n': n
    }

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
    
    ct = cor_test(final_model, final_exp, method='pearson')
    r = ct['correlation']
    p = ct['p_value']
    lo, hi = ct['conf_int']

    # NOVO: teste de CCC (Lin, 1989)
    ct_ccc = ccc_test(final_model, final_exp)
    c_val = ct_ccc['ccc']; p_c = ct_ccc['p_value']; lo_c, hi_c = ct_ccc['conf_int']

        # também manda pro console, se quiser “printar o teste”
    print("=== cor.test (Pearson) ===")
    print(f"n = {ct['n']}, r = {r:.6f}, p = {p:.6g}, CI95% = {ct['conf_int']}")
    
    print("=== ccc.test (Lin, 1989) ===")
    print(f"n = {ct_ccc['n']}, CCC = {c_val:.6f}, p = {p_c:.6g}, CI95% = {ct_ccc['conf_int']}")
    if save:
        plt.savefig(figure_name + '_sp.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close()
    return



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
