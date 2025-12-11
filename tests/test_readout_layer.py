import unittest
import numpy as np

from reservoir_lm.readout_layer.layer import ReadoutLayer
from reservoir_lm.readout_layer.trainer import Trainer
from reservoir_lm.readout_layer.predictor import Predictor

class TestReadoutLayer(unittest.TestCase):
    """Unit tests for the readout_layer module."""

    def setUp(self):
        self.n_reservoir = 50
        self.output_dim = 10
        self.seq_len = 5
        self.batch_size = 2

        self.readout = ReadoutLayer(self.n_reservoir, self.output_dim)
        self.states = np.random.rand(self.batch_size, self.seq_len, self.n_reservoir)

    def test_readout_layer_with_bias(self):
        """Test the ReadoutLayer class with a bias term."""
        self.readout = ReadoutLayer(self.n_reservoir, self.output_dim, add_bias=True)
        self.assertEqual(self.readout.W_out.shape, (self.n_reservoir + 1, self.output_dim))

        output = self.readout.transform(self.states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))

    def test_readout_layer_no_bias(self):
        """Test the ReadoutLayer class without a bias term."""
        self.readout = ReadoutLayer(self.n_reservoir, self.output_dim, add_bias=False)
        self.assertEqual(self.readout.W_out.shape, (self.n_reservoir, self.output_dim))

        output = self.readout.transform(self.states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))

    def test_trainer_with_bias(self):
        """Test the Trainer class with a bias term."""
        readout = ReadoutLayer(self.n_reservoir, self.output_dim, add_bias=True)
        trainer = Trainer(readout, method="ridge", alpha=0.1)

        X = self.states.reshape(-1, self.n_reservoir)
        Y = np.random.rand(self.batch_size * self.seq_len, self.output_dim)

        trainer.train(X, Y)
        self.assertFalse(np.all(readout.W_out == 0))
        self.assertEqual(readout.W_out.shape, (self.n_reservoir + 1, self.output_dim))

    def test_trainer_no_bias(self):
        """Test the Trainer class without a bias term."""
        readout = ReadoutLayer(self.n_reservoir, self.output_dim, add_bias=False)
        trainer = Trainer(readout, method="ridge", alpha=0.1)

        X = self.states.reshape(-1, self.n_reservoir)
        Y = np.random.rand(self.batch_size * self.seq_len, self.output_dim)

        trainer.train(X, Y)
        self.assertFalse(np.all(readout.W_out == 0))
        self.assertEqual(readout.W_out.shape, (self.n_reservoir, self.output_dim))

    def test_predictor_with_bias(self):
        """Test the Predictor class with a bias term."""
        readout = ReadoutLayer(self.n_reservoir, self.output_dim, add_bias=True)
        W_out = np.random.rand(self.n_reservoir + 1, self.output_dim)
        readout.set_weights(W_out)

        predictor = Predictor(readout)
        probabilities = predictor.predict(self.states)

        self.assertEqual(probabilities.shape, (self.batch_size, self.seq_len, self.output_dim))

        sums = np.sum(probabilities, axis=-1)
        np.testing.assert_array_almost_equal(sums, np.ones((self.batch_size, self.seq_len)))

    def test_predictor_no_bias(self):
        """Test the Predictor class without a bias term."""
        readout = ReadoutLayer(self.n_reservoir, self.output_dim, add_bias=False)
        W_out = np.random.rand(self.n_reservoir, self.output_dim)
        readout.set_weights(W_out)

        predictor = Predictor(readout)
        probabilities = predictor.predict(self.states)

        self.assertEqual(probabilities.shape, (self.batch_size, self.seq_len, self.output_dim))

        sums = np.sum(probabilities, axis=-1)
        np.testing.assert_array_almost_equal(sums, np.ones((self.batch_size, self.seq_len)))

if __name__ == "__main__":
    unittest.main()
