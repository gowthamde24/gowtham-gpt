import torch
import torch.nn as nn
from torchtyping import TensorType


class SentimentClassifier(nn.Module):
    """Bag-of-embeddings sentiment classifier: Embedding -> mean-pool -> Linear(16, 1) -> Sigmoid."""

    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        """Predicts a per-example sentiment probability from a batch of token-id sequences."""
        embeddings = self.embedding_layer(x)
        averaged = torch.mean(embeddings, dim=1)
        projected = self.linear(averaged)
        sigmoid_output = self.sigmoid_layer(projected)
        return torch.round(sigmoid_output, decimals=4)
