import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


def safe_pearsonr(y_true, y_pred):
    """Pearson correlation that returns 0.0 instead of NaN when one of the
    two vectors is constant (undefined correlation)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "pearson_r": safe_pearsonr(y_true, y_pred),
    }
