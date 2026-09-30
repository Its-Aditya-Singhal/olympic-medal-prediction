"""Loss functions from the paper (Section 4.1)."""
import numpy as np


def mse(y_true, y_pred) -> float:
    """Eq. 1: average squared loss over all countries."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_pred - y_true) ** 2))


def top10_mse(y_true, y_pred, k: int = 10) -> float:
    """Eq. 2: squared loss restricted to the k countries with the most actual medals.

    The paper writes the indicator inside a 1/n average; we average over the
    top-k countries instead so the number is on the same scale as Eq. 1.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    top = np.argsort(-y_true)[:k]
    return float(np.mean((y_pred[top] - y_true[top]) ** 2))


def combined(y_true, y_pred) -> float:
    """Model selection objective (Section 6): Eq.1 + 0.25 * Eq.2."""
    return mse(y_true, y_pred) + 0.25 * top10_mse(y_true, y_pred)


def rmse(y_true, y_pred) -> float:
    """The paper's "average standard deviation" = sqrt of Eq. 1."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def report(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "top10_rmse": float(np.sqrt(top10_mse(y_true, y_pred))),
        "combined": combined(y_true, y_pred),
    }
