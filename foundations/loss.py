import numpy as np
from numpy.typing import NDArray


def binary_cross_entropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Mean binary cross-entropy loss between true labels (0/1) and predicted probabilities."""
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # avoid log(0)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return round(loss, 4)


def categorical_cross_entropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Mean categorical cross-entropy loss between one-hot labels and predicted class probabilities."""
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # avoid log(0)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return round(loss, 4)
