import numpy as np

__all__ = [
    "max_error",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "median_absolute_error",
    "mean_absolute_percentage_error",
    "r2_score",
    "root_mean_squared_log_error",
    "root_mean_squared_error",
    "explained_variance_score",
]

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_squared_log_error(y_true, y_pred):
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Mean Squared Log Error cannot be used when targets contain negative values.")
    
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)


def median_absolute_error(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined for targets with zero values.")
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def max_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def explained_variance_score(y_true, y_pred):
    numerator = np.var(y_true - y_pred)
    denominator = np.var(y_true)
    return 1 - numerator / denominator if denominator != 0 else 0.0
