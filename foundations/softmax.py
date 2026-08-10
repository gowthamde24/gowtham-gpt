import numpy as np
from numpy.typing import NDArray


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Softmax over a 1D array of logits, converting them to a probability distribution."""
    shifted = z - np.max(z)  # subtract max for numerical stability
    exps = np.exp(shifted)
    return np.round(exps / np.sum(exps), 4)
