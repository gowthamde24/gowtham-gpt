"""KV-caching: reuse past keys/values during autoregressive generation instead of
recomputing attention over the whole prefix at every step.

This is an advanced/optional component — the main GPT model in `gpt.py` does not use
it. It's provided as a standalone building block for a faster generation loop, where
each step only needs to project the *new* token instead of the full context.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class KVCache:
    """Holds the running (key, value) tensors for one attention layer across steps."""

    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append the newest step's K/V and return the full cached K/V so far."""
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    """Single-head attention that reads/writes through a `KVCache` instead of
    recomputing K/V for tokens it has already seen."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if kv_cache is None:
            kv_cache = KVCache()
        full_k, full_v = kv_cache.update(k, v)

        scores = (q @ full_k.transpose(-2, -1)) * (full_k.shape[-1] ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ full_v

        return output, kv_cache
