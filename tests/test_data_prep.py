import unittest
import numpy as np

from reservoir_lm.data_prep.tokenizer import Tokenizer
from reservoir_lm.data_prep.encoder import TimeSeriesEncoder
from reservoir_lm.data_prep.loader import SequenceDataLoader


class TestDataPrep(unittest.TestCase):
    """Unit tests for the data_prep module."""

    def setUp(self):
        """Set up common resources for tests."""
        self.texts = ["a simple sentence", "another sentence"]

    def test_tokenizer(self):
        """Test the Tokenizer class."""
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.texts)

        self.assertEqual(tokenizer.vocab_size, 6)  # 4 unique words + pad/unk
        self.assertIn("simple", tokenizer.word_index)

        sequences = tokenizer.texts_to_sequences(["a new sentence"])
        self.assertEqual(len(sequences[0]), 3)
        self.assertEqual(sequences[0][1], tokenizer.word_index[tokenizer.unk_token])

        restored_texts = tokenizer.sequences_to_texts(sequences)
        self.assertEqual(restored_texts[0], "a <unk> sentence")

    def test_time_series_encoder(self):
        """Test the TimeSeriesEncoder class."""
        vocab_size = 10
        embedding_dim = 20
        encoder = TimeSeriesEncoder(vocab_size, embedding_dim)
        self.assertEqual(encoder.embedding.shape, (vocab_size, embedding_dim))

        sequences = [[1, 2, 3], [4, 5]]
        encoded = encoder.encode(sequences)
        self.assertEqual(len(encoded), 2)
        self.assertEqual(encoded[0].shape, (3, embedding_dim))
        self.assertIsInstance(encoded[0], np.ndarray)

    def test_sequence_data_loader(self):
        """Test the SequenceDataLoader class."""
        sequences = [[i for i in range(10)]]
        seq_len = 4
        batch_size = 2

        loader = SequenceDataLoader(sequences, sequence_length=seq_len, batch_size=batch_size)

        # Total items: 10 - 4 = 6
        # Batches: ceil(6 / 2) = 3
        self.assertEqual(len(loader), 3)

        batch_count = 0
        for inputs, targets in loader:
            self.assertIn(inputs.shape[0], [1, 2])
            self.assertEqual(inputs.shape[1], seq_len)
            self.assertEqual(targets.shape, inputs.shape)
            # Check that target is input shifted by one
            np.testing.assert_array_equal(inputs[0, 1:], targets[0, :-1])
            batch_count += 1

        self.assertEqual(batch_count, 3)

if __name__ == "__main__":
    unittest.main()
