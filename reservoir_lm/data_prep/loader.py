import numpy as np
from typing import List, Iterator, Tuple


class SequenceDataLoader:
    """
    Splits long text into fixed-length sequences and provides an iterator for batch processing.
    """

    def __init__(
        self,
        sequences: List[List[int]],
        sequence_length: int,
        batch_size: int,
        stride: int = 1,
    ):
        """
        Args:
            sequences (List[List[int]]): A list of token ID sequences.
            sequence_length (int): The length of each input sequence chunk.
            batch_size (int): The number of sequences per batch.
            stride (int): The step size to move for creating the next sequence.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.stride = stride

        # Flatten the list of sequences into a single sequence
        flat_sequence = [token for seq in sequences for token in seq]

        self.inputs: List[List[int]] = []
        self.targets: List[List[int]] = []

        self._create_io_pairs(flat_sequence)

        self.num_batches = int(np.ceil(len(self.inputs) / self.batch_size))

    def _create_io_pairs(self, sequence: List[int]):
        """Creates input and target pairs from a long sequence."""
        for i in range(0, len(sequence) - self.sequence_length, self.stride):
            self.inputs.append(sequence[i : i + self.sequence_length])
            self.targets.append(sequence[i + 1 : i + self.sequence_length + 1])

    def __len__(self) -> int:
        """Returns the number of batches."""
        return self.num_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Returns an iterator that yields batches of inputs and targets."""
        for i in range(0, len(self.inputs), self.batch_size):
            batch_inputs = self.inputs[i : i + self.batch_size]
            batch_targets = self.targets[i : i + self.batch_size]

            yield np.array(batch_inputs), np.array(batch_targets)
