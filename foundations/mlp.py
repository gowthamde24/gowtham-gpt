import numpy as np
from numpy.typing import NDArray
from typing import List


def mlp_forward(x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Forward pass through a multi-layer perceptron, ReLU on hidden layers and none on the output layer."""
    h = x
    for i in range(len(weights)):
        h = h @ weights[i] + biases[i]
        if i < len(weights) - 1:
            h = np.maximum(0, h)
    return np.round(h, 5)
