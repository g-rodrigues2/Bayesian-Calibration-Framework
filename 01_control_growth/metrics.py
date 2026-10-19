"""Agreement metrics between data and model predictions."""
import numpy as np


def ccc(x, y):
    """Concordance Correlation Coefficient (Lin, 1989)."""
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
    return (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)


def pcc(x, y):
    """Pearson Correlation Coefficient."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must not be empty")
    std_x, std_y = np.std(x), np.std(y)
    if std_x == 0 or std_y == 0:
        raise ValueError("Input arrays must not have zero variance")
    covariance = np.cov(x, y, bias=True)[0, 1]
    return covariance / (std_x * std_y)


def nrmse(actual, pred):
    """Normalized root mean squared error (%)."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    if len(actual) != len(pred):
        raise ValueError("Input arrays must have the same length")
    if actual.size == 0:
        raise ValueError("Input arrays must not be empty")
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return rmse / np.mean(actual) * 100


def mape(pred, actual):
    """Mean absolute percentage error (%)."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    if len(actual) != len(pred):
        raise ValueError("Input arrays must have the same length")
    if actual.size == 0:
        raise ValueError("Input arrays must not be empty")
    if np.any(actual == 0):
        raise ValueError("Actual values must not contain zeros")
    return np.mean(np.abs((actual - pred) / actual)) * 100
