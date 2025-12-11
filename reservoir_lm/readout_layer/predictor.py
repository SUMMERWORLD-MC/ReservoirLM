import numpy as np
from .layer import ReadoutLayer


class Predictor:
    """
    Uses a trained ReadoutLayer to compute the probability distribution of the next token.
    """

    def __init__(self, readout_layer: ReadoutLayer):
        """
        Args:
            readout_layer (ReadoutLayer): A trained readout layer with the learned W_out.
        """
        self.readout_layer = readout_layer

    def predict(self, reservoir_states: np.ndarray) -> np.ndarray:
        """
        Computes the next-token probability distribution from the reservoir states.

        Args:
            reservoir_states (np.ndarray): The reservoir's state trajectory, with shape
                                           (batch_size, sequence_length, n_reservoir).

        Returns:
            np.ndarray: The probability distributions over the vocabulary, with shape
                        (batch_size, sequence_length, output_dim).
        """
        logits = self.readout_layer.transform(reservoir_states)
        return self._softmax(logits)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the softmax function, stabilizing the computation by subtracting the max.
        """
        # Subtract the max for numerical stability
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
