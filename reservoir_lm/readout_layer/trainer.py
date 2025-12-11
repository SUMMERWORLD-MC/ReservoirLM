import numpy as np
from .layer import ReadoutLayer


class Trainer:
    """
    Trains the ReadoutLayer by directly computing the optimal output weights (W_out)
    using a fast linear solver.
    """

    def __init__(self, readout_layer: ReadoutLayer, method: str = "ridge", alpha: float = 1e-4):
        """
        Args:
            readout_layer (ReadoutLayer): The readout layer to be trained.
            method (str): The solver method ('ridge' or 'pinv').
            alpha (float): The regularization strength for ridge regression.
        """
        self.readout_layer = readout_layer
        self.method = method
        self.alpha = alpha

    def _add_bias_to_states(self, X: np.ndarray) -> np.ndarray:
        """Adds a bias term (column of ones) to the input matrix X."""
        biases = np.ones((X.shape[0], 1))
        return np.concatenate([X, biases], axis=1)

    def train(self, X: np.ndarray, Y: np.ndarray):
        """
        Trains the readout weights W_out using the given reservoir states and target outputs.

        Args:
            X (np.ndarray): The reservoir state trajectory, with shape
                            (batch_size * sequence_length, n_reservoir).
            Y (np.ndarray): The target outputs, with shape
                            (batch_size * sequence_length, output_dim).
        """
        if self.readout_layer.add_bias:
            X = self._add_bias_to_states(X)

        if self.method == "ridge":
            # Ridge Regression: W_out = (X^T @ X + alpha * I)^-1 @ X^T @ Y
            identity = np.identity(X.shape[1])
            A = X.T @ X + self.alpha * identity
            B = X.T @ Y
            W_out = np.linalg.solve(A, B)
        elif self.method == "pinv":
            # Moore-Penrose Pseudoinverse: W_out = pinv(X) @ Y
            W_out = np.linalg.pinv(X) @ Y
        else:
            raise ValueError(f"Unknown training method: {self.method}")

        self.readout_layer.set_weights(W_out)
