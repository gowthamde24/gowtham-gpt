import numpy as np
from numpy.typing import NDArray


def predict(X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear model prediction: X @ weights."""
    prediction = np.matmul(X, weights)
    return np.round(prediction, 5)


def mean_squared_error(prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
    """Mean squared error between predictions and ground truth."""
    error = np.mean(np.square(prediction - ground_truth))
    return round(error, 5)
