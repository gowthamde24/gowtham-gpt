import numpy as np
from numpy.typing import NDArray


def neuron_forward(x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
    """Forward pass of a single neuron: activation(dot(x, w) + b)."""
    z = np.dot(x, w) + b

    if activation == "sigmoid":
        result = 1.0 / (1.0 + np.exp(-z))
    elif activation == "relu":
        result = max(0.0, z)
    else:
        result = z

    return np.round(result, 5)
