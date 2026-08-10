import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def train_linear_regression(X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
    """Fits a linear regression model (y_hat = X @ w + b) via full-batch gradient descent on MSE."""
    n = X.shape[0]

    w = np.zeros(X.shape[1])
    b = 0.0

    for i in range(epochs):
        y_hat = X @ w + b
        error = y_hat - y

        dw = (2.0 / n) * (X.T @ error)
        db = (2.0 / n) * np.sum(error)

        w = w - lr * dw
        b = b - lr * db

    return (np.round(w, 5), round(float(b), 5))
