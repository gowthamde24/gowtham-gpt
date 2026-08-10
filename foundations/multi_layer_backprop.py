import numpy as np
from typing import List


def two_layer_forward_backward(x: List[float],
                                W1: List[List[float]], b1: List[float],
                                W2: List[List[float]], b2: List[float],
                                y_true: List[float]) -> dict:
    """Runs a forward and backward pass through a Linear -> ReLU -> Linear network under MSE loss.

    Returns a dict with the loss and the gradients for W1, b1, W2, b2.
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    b1 = np.array(b1)
    b2 = np.array(b2)
    y_true = np.array(y_true)

    # forward pass
    z1 = x @ W1.T + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2.T + b2
    loss = np.mean((z2 - y_true) ** 2)

    # backward pass
    n = len(y_true) if y_true.ndim > 0 else 1
    dz2 = 2 * (z2 - y_true) / n
    dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)
    db2 = dz2

    da1 = dz2.reshape(-1, 1) @ W2
    da1 = da1.flatten()
    dz1 = da1 * (z1 > 0).astype(float)
    dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
    db1 = dz1

    return {
        'loss': round(float(loss), 4),
        'dW1': np.round(dW1, 4).tolist(),
        'db1': np.round(db1, 4).tolist(),
        'dW2': np.round(dW2, 4).tolist(),
        'db2': np.round(db2, 4).tolist(),
    }
