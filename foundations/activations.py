import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Elementwise logistic sigmoid: 1 / (1 + e^-z)."""
    return np.round(1 / (1 + np.exp(-z)), 5)


def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Elementwise rectified linear unit: max(0, z)."""
    return np.maximum(0, z)
