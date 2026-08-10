"""From-scratch neural-network building blocks built while completing the NeetCode ML course."""

# Gradient descent & optimization
from .gradient_descent import minimize_quadratic
from .linear_regression import predict, mean_squared_error
from .linear_regression_training import LinearRegressionTrainer
from .training_loop import train_linear_regression

# Neural network primitives
from .activations import sigmoid, relu
from .softmax import softmax
from .loss import binary_cross_entropy, categorical_cross_entropy
from .neuron import neuron_forward
from .backprop import neuron_backward
from .multi_layer_backprop import two_layer_forward_backward
from .mlp import mlp_forward
from .weight_init import xavier_init, kaiming_init, check_activations

# PyTorch basics
from .pytorch_basics import reshape_pairs, column_mean, concat_columns, mse_loss
from .digit_classifier import DigitClassifier
from .sentiment import SentimentClassifier

# Diagnostics
from .training_diagnostics import TrainingDiagnostics
from .dead_relu_detector import DeadReLUDetector
