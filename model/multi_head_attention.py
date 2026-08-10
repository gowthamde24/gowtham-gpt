"""Multi-headed causal self-attention: several attention heads run in parallel and merged."""

import torch
import torch.nn as nn
from torchtyping import TensorType

from .attention import SingleHeadAttention


class MultiHeadSelfAttention(nn.Module):
    """Runs `num_heads` independent attention heads and projects their concatenated output."""

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        head_size = attention_dim // num_heads
        self.att_heads = nn.ModuleList(
            SingleHeadAttention(embedding_dim, head_size) for _ in range(num_heads)
        )
        self.output_proj = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        head_outputs = [head(embedded) for head in self.att_heads]
        concatenated = torch.cat(head_outputs, dim=2)
        return self.output_proj(concatenated)
