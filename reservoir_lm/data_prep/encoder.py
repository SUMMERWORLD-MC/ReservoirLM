import numpy as np
from typing import List


class TimeSeriesEncoder:
    """Encodes token IDs into time-series embedding vectors."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.1

    def encode(self, sequences: List[List[int]]) -> List[np.ndarray]:
        """Converts sequences of token IDs into embedding vectors."""
        encoded_sequences = []
        for sequence in sequences:
            embedded_sequence = self.embedding[sequence]
            encoded_sequences.append(embedded_sequence)
        return encoded_sequences

    def get_embedding_matrix(self) -> np.ndarray:
        """Returns the embedding matrix."""
        return self.embedding
