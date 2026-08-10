import numpy as np
from numpy.typing import NDArray


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> NDArray[np.float64]:
    """Sinusoidal positional encoding table of shape (seq_len, d_model), as in "Attention Is All You Need".

    This is a from-scratch numpy reference; the production GPT model uses learned position
    embeddings instead of sinusoidal ones.
    """
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
    PE[:, 0::2] = np.sin(position / div_term)
    PE[:, 1::2] = np.cos(position / div_term[:PE[:, 1::2].shape[1]])
    return np.round(PE, 5)
