import numpy as np
from numpy.typing import NDArray


def embedding_lookup(embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
    """Looks up embedding vectors for a batch of token IDs from an (vocab_size, embed_dim) table."""
    return np.round(embeddings[token_ids], 5)
