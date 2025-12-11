import numpy as np
from typing import Callable


class ActivationFunction:
    """Provides non-linear activation functions for reservoir nodes."""

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU) activation function."""
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def get(name: str) -> Callable[[np.ndarray], np.ndarray]:
        """Returns the specified activation function."""
        if name == "tanh":
            return ActivationFunction.tanh
        elif name == "relu":
            return ActivationFunction.relu
        elif name == "sigmoid":
            return ActivationFunction.sigmoid
        else:
            raise ValueError(f"Unknown activation function: {name}")
