import numpy as np
from typing import List


def rms_norm(x: List[float], gamma: List[float], eps: float) -> List[float]:
    """RMS-normalizes a vector (LayerNorm without mean-centering or a shift term), then scales by gamma."""
    x = np.array(x)
    gamma = np.array(gamma)

    rms = np.sqrt(np.mean(x ** 2) + eps)
    x_hat = x / rms

    return np.round(gamma * x_hat, 4).tolist()
