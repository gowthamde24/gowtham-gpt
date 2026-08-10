import torch
import torch.nn
from torchtyping import TensorType


def reshape_pairs(to_reshape: TensorType[float]) -> TensorType[float]:
    """Reshapes an (M, N) tensor into (M*N/2, 2) pairs."""
    M, N = to_reshape.shape
    reshaped = torch.reshape(to_reshape, (M * N // 2, 2))
    return torch.round(reshaped, decimals=4)


def column_mean(to_avg: TensorType[float]) -> TensorType[float]:
    """Column-wise mean (average across rows)."""
    averaged = torch.mean(to_avg, dim=0)
    return torch.round(averaged, decimals=4)


def concat_columns(cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
    """Joins two tensors side-by-side along dim=1."""
    concatenated = torch.cat((cat_one, cat_two), dim=1)
    return torch.round(concatenated, decimals=4)


def mse_loss(prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
    """Mean squared error between prediction and target."""
    loss = torch.nn.functional.mse_loss(prediction, target)
    return torch.round(loss, decimals=4)
