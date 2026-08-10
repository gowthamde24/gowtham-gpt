import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


def build_sentiment_dataset(positive: List[str], negative: List[str]) -> TensorType[float]:
    """Builds a padded, word-id-encoded tensor dataset from positive and negative sentences."""
    combined = []
    for sentence in positive:
        combined.append(sentence)
    for sentence in negative:
        combined.append(sentence)

    unique_words = set()
    for sentence in combined:
        words = sentence.split()
        for word in words:
            unique_words.add(word)

    vacabulary = sorted(unique_words)

    word_to_id = {}
    next_id = 1
    for word in vacabulary:
        word_to_id[word] = next_id
        next_id += 1

    encoded = []
    for sentence in combined:
        ids = []
        words = sentence.split()
        for word in words:
            ids.append(word_to_id[word])
        encoded.append(torch.tensor(ids))

    padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True)
    return padded
