"""Small helpers shared across train.py, generate.py, and the web dashboard."""

import torch


def pick_device() -> torch.device:
    """Best available device: CUDA > Apple Silicon MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
