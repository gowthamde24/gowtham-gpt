"""Random batch sampling for next-token-prediction training."""

from typing import Optional, Tuple

import torch
from torchtyping import TensorType


def get_batch(
    data: TensorType[int],
    context_length: int,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[TensorType[int], TensorType[int]]:
    """Sample `batch_size` random windows of `context_length` tokens from `data`.

    Returns (X, Y) where Y is X shifted one position to the right — the standard
    next-token-prediction training pair. Pass a `torch.Generator` for reproducible
    sampling; omit it to get fresh random batches each call, as real training needs.
    """
    start_indices = torch.randint(
        len(data) - context_length, (batch_size,), generator=generator
    )
    x = torch.stack([data[i : i + context_length] for i in start_indices])
    y = torch.stack([data[i + 1 : i + 1 + context_length] for i in start_indices])
    return x, y
