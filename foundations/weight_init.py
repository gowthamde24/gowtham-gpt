import torch
import math
from typing import List


def xavier_init(fan_in: int, fan_out: int) -> List[List[float]]:
    """Xavier/Glorot-normal-initialized (fan_out x fan_in) weight matrix."""
    torch.manual_seed(0)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    weights = torch.randn(fan_out, fan_in) * std
    return torch.round(weights, decimals=4).tolist()


def kaiming_init(fan_in: int, fan_out: int) -> List[List[float]]:
    """Kaiming/He-normal-initialized (fan_out x fan_in) weight matrix, for ReLU networks."""
    torch.manual_seed(0)
    std = math.sqrt(2.0 / (fan_in))
    weights = torch.randn(fan_out, fan_in) * std
    return torch.round(weights, decimals=4).tolist()


def check_activations(num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
    """Standard deviation of activations after each ReLU layer of a randomly initialized MLP.

    Useful for observing how xavier/kaiming/default init affect activation scale with depth.
    """
    torch.manual_seed(0)
    dims = [input_dim] + [hidden_dim] * num_layers
    weights = []
    for i in range(num_layers):
        if init_type == 'xavier':
            std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
        elif init_type == 'kaiming':
            std = math.sqrt(2.0 / dims[i])
        else:
            std = 1.0
        w = torch.randn(dims[i + 1], dims[i]) * std
        weights.append(w)

    x = torch.randn(1, input_dim)
    stds = []
    for w in weights:
        x = x @ w.T
        x = torch.relu(x)
        stds.append(round(x.std().item(), 2))
    return stds
