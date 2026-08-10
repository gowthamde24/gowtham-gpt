"""Causal self-attention: the core primitive every block in the GPT is built from."""

import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    """One causal attention head: Q/K/V projections + masked scaled dot-product attention."""

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        attention_dim = k.shape[2]
        scores = scores / (attention_dim ** 0.5)

        # Causal mask: position i may only attend to positions <= i.
        context_length = k.shape[1]
        causal_mask = torch.tril(torch.ones(context_length, context_length, device=embedded.device))
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        scores = nn.functional.softmax(scores, dim=-1)

        return scores @ v
