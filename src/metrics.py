import numpy as np


def wape(y, p):
    """Weighted APE. В отличие от MAPE не взрывается на нулях,
    а их тут ~8% (ночные часы на окраинах)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return np.abs(y - p).sum() / y.sum()


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def bias(y, p):
    """>0 значит модель в среднем завышает."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return (p.sum() - y.sum()) / y.sum()


def report(y, p) -> dict:
    return {"WAPE": wape(y, p), "MAE": mae(y, p), "RMSE": rmse(y, p), "bias": bias(y, p)}
