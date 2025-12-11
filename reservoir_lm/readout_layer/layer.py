import numpy as np


class ReadoutLayer:
    """
    Holds the output weight matrix W_out and performs a linear transformation from
    reservoir states to the desired output (e.g., vocabulary-sized logits).
    """

    def __init__(self, n_reservoir: int, output_dim: int, add_bias: bool = True):
        """
        Args:
            n_reservoir (int): The number of reservoir units.
            output_dim (int): The dimensionality of the output (e.g., vocab size).
            add_bias (bool): If True, a bias term is concatenated to the reservoir states.
        """
        self.n_reservoir = n_reservoir
        self.output_dim = output_dim
        self.add_bias = add_bias

        # Adjust feature dimension to account for the bias term
        n_features = self.n_reservoir + 1 if self.add_bias else self.n_reservoir

        # Initialize W_out with a placeholder; it will be trained later.
        self.W_out = np.zeros((n_features, output_dim))

    def set_weights(self, W_out: np.ndarray):
        """Sets the output weight matrix."""
        if W_out.shape != self.W_out.shape:
            raise ValueError("Incorrect shape for W_out.")
        self.W_out = W_out

    def _add_bias_to_states(self, reservoir_states: np.ndarray) -> np.ndarray:
        """Concatenates a bias term (a column of ones) to the reservoir states."""
        batch_size, seq_len, _ = reservoir_states.shape
        biases = np.ones((batch_size, seq_len, 1))
        return np.concatenate([reservoir_states, biases], axis=-1)

    def transform(self, reservoir_states: np.ndarray) -> np.ndarray:
        """
        Applies the linear transformation to the reservoir states.

        Args:
            reservoir_states (np.ndarray): The reservoir's state trajectory, with shape
                                           (batch_size, sequence_length, n_reservoir).

        Returns:
            np.ndarray: The output logits, with shape
                        (batch_size, sequence_length, output_dim).
        """
        if self.add_bias:
            reservoir_states = self._add_bias_to_states(reservoir_states)
        return np.dot(reservoir_states, self.W_out)
