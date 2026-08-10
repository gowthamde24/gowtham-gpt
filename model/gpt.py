"""The full GPT model: embeddings -> stacked transformer blocks -> vocabulary projection."""

import torch
import torch.nn as nn
from torchtyping import TensorType

from .transformer import TransformerBlock


class GPT(nn.Module):
    """A decoder-only transformer language model.

    Returns raw logits (not probabilities) of shape (batch, seq_len, vocab_size) — the
    caller is responsible for softmax (sampling) or cross-entropy (training).
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.2,
        seed: int | None = 0,
    ):
        super().__init__()
        if seed is not None:
            # Seed once, here, so weight init is reproducible without forcing every
            # submodule to start from the same RNG state (that would make every
            # attention head/layer identical at init and break symmetry).
            torch.manual_seed(seed)

        self.context_length = context_length
        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.position_embeddings = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential(
            *(TransformerBlock(model_dim, num_heads, dropout) for _ in range(num_blocks))
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        positions = torch.arange(context.shape[1], device=context.device)
        embedded = self.word_embeddings(context) + self.position_embeddings(positions)

        output = self.final_norm(self.transformer_blocks(embedded))
        return self.vocab_projection(output)
