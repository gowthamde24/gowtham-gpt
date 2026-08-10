"""GPT architecture: production components plus optional/advanced building blocks.

Production path (what `train.py` / `generate.py` use):
    GPT -> TransformerBlock -> MultiHeadSelfAttention -> SingleHeadAttention

Optional building blocks, not wired into `GPT` by default:
    KVCache / CachedAttention   — cached autoregressive generation
    GroupedQueryAttention       — cheaper attention for larger context windows

Reference implementations (from-scratch, numpy, kept for learning — the production
model above uses `torch.nn.LayerNorm` / `nn.Embedding` directly instead):
    layer_norm, batch_norm, rms_norm, embedding_lookup, sinusoidal_positional_encoding
"""

from .attention import SingleHeadAttention
from .multi_head_attention import MultiHeadSelfAttention
from .transformer import FeedForward, TransformerBlock
from .gpt import GPT
from .kv_cache import KVCache, CachedAttention
from .grouped_query_attention import GroupedQueryAttention
from .normalization import layer_norm
from .batch_normalization import batch_norm
from .rms_normalization import rms_norm
from .embeddings import embedding_lookup
from .positional_encoding import sinusoidal_positional_encoding

__all__ = [
    "SingleHeadAttention",
    "MultiHeadSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "GPT",
    "KVCache",
    "CachedAttention",
    "GroupedQueryAttention",
    "layer_norm",
    "batch_norm",
    "rms_norm",
    "embedding_lookup",
    "sinusoidal_positional_encoding",
]
