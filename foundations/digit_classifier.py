import torch
import torch.nn as nn
from torchtyping import TensorType


class DigitClassifier(nn.Module):
    """MNIST-style digit classifier: Linear(784, 512) -> ReLU -> Dropout -> Linear(512, 10) -> Sigmoid."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first_linear = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        """Predicts per-class scores for a batch of flattened (batch_size, 784) images."""
        torch.manual_seed(0)
        x = self.first_linear(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = self.sigmoid(x)
        return torch.round(x, decimals=4)
