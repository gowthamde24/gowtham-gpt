import torch
import torch.nn as nn
from typing import List


class DeadReLUDetector:
    """Detects dead ReLU neurons in a model and suggests a fix based on their pattern."""

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        """Fraction of neurons that output 0 for every sample in the batch, per ReLU layer."""
        dead_fractions = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.ReLU):
                    dead = (x == 0).all(dim=0).float().mean().item()
                    dead_fractions.append(round(dead, 4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        """Recommends a remedy based on the per-layer dead-neuron fractions.

        Checks, in order: any layer > 0.5 dead -> use_leaky_relu; first layer > 0.3 dead ->
        reinitialize; dead fraction strictly increasing with depth and last layer > 0.1 ->
        reduce_learning_rate; otherwise healthy.
        """
        if len(dead_fractions) == 0:
            return 'healthy'
        max_frac = max(dead_fractions)
        if max_frac > 0.5:
            return 'use_leaky_relu'

        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        if len(dead_fractions) >= 2:
            increasing = all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )
            if increasing and dead_fractions[-1] > 0.1:
                return 'reduce_learning_rate'
        if max_frac < 0.1:
            return 'healthy'

        return 'healthy'
