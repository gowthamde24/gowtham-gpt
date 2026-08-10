"""A single transformer block: attention + feed-forward, each wrapped in a Pre-LN residual."""

import torch.nn as nn
from torchtyping import TensorType

from .multi_head_attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, ReLU, project back down, dropout for regularization."""

    def __init__(self, model_dim: int, dropout: float = 0.2):
        super().__init__()
        self.up_projection = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        return self.dropout(self.down_projection(self.relu(self.up_projection(x))))


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LayerNorm is applied *before* each sub-layer, not after.

    This differs from the original "Attention Is All You Need" diagram (Post-LN) but
    trains more stably, which is why virtually every modern GPT-style model uses it.
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.attention = MultiHeadSelfAttention(model_dim, model_dim, num_heads)
        self.feed_forward = FeedForward(model_dim, dropout)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        embedded = embedded + self.attention(self.first_norm(embedded))
        embedded = embedded + self.feed_forward(self.second_norm(embedded))
        return embedded
