"""Grouped-query attention (GQA): fewer K/V heads than Q heads, each K/V head shared by
a group of query heads. Cuts the K/V cache size (and memory bandwidth) at inference
time with minimal quality loss versus full multi-head attention.

This is an advanced/optional component — the main GPT model in `gpt.py` uses standard
multi-head attention. Swap it in for `MultiHeadSelfAttention` if you need larger
context windows or cheaper inference.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Each KV head is shared by (num_heads // num_kv_heads) query heads.
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.output_proj(out)
