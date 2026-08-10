import numpy as np
from numpy.typing import NDArray


def layer_norm(x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Layer-normalizes a 1D feature vector, then applies a learned scale (gamma) and shift (beta).

    This is a from-scratch numpy reference for learning purposes; the production GPT model
    uses torch.nn.LayerNorm directly for performance.
    """
    eps = 1e-5
    mean = np.mean(x)
    var = np.var(x)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    return np.round(out, 5)
