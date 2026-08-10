import torch
from typing import List, Tuple


def build_word_batches(raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
    """Samples random (X, Y) next-word-prediction batches from whitespace-tokenized text.

    This is a word-level batching reference; the main training pipeline instead uses
    character-level batching in data/loader.py.
    """
    torch.manual_seed(0)

    tokenized = raw_dataset.split()
    indices = torch.randint(low=0, high=len(tokenized) - context_length, size=(batch_size,)).tolist()
    X = []
    Y = []
    for idx in indices:
        X.append(tokenized[idx:idx + context_length])
        Y.append(tokenized[idx + 1:idx + 1 + context_length])
    return X, Y
