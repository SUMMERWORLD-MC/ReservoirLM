import unittest
import numpy as np
from scipy.sparse import issparse

from reservoir_lm.reservoir_core.cell import ReservoirCell
from reservoir_lm.reservoir_core.simulator import ReservoirSimulator
from reservoir_lm.reservoir_core.activation import ActivationFunction

class TestReservoirCore(unittest.TestCase):
    """Unit tests for the reservoir_core module."""

    def setUp(self):
        self.n_reservoir = 100
        self.n_input = 20

    def test_reservoir_cell(self):
        """Test the ReservoirCell class."""
        cell = ReservoirCell(self.n_reservoir, self.n_input, spectral_radius=0.9)

        self.assertEqual(cell.W_in.shape, (self.n_reservoir, self.n_input))
        self.assertEqual(cell.W.shape, (self.n_reservoir, self.n_reservoir))
        self.assertEqual(cell.W_bias.shape, (self.n_reservoir,))

        # Check if spectral radius is close to the target
        W_dense = cell.W.toarray() if issparse(cell.W) else cell.W
        eigenvalues, _ = np.linalg.eig(W_dense)
        spectral_radius = np.max(np.abs(eigenvalues))
        self.assertAlmostEqual(spectral_radius, 0.9, places=5)

    def test_reservoir_simulator(self):
        """Test the ReservoirSimulator class."""
        cell = ReservoirCell(self.n_reservoir, self.n_input)
        simulator = ReservoirSimulator(cell, activation="tanh")

        seq_len = 10
        input_sequence = np.random.rand(seq_len, self.n_input)

        # Test without warmup
        states = simulator.simulate(input_sequence, warmup_steps=0)
        self.assertEqual(states.shape, (seq_len, self.n_reservoir))

        # Test with warmup
        warmup = 2
        states_warmup = simulator.simulate(input_sequence, warmup_steps=warmup)
        self.assertEqual(states_warmup.shape, (seq_len - warmup, self.n_reservoir))

    def test_activation_function(self):
        """Test the ActivationFunction class."""
        x = np.array([-1, 0, 1])

        # Tanh
        tanh_func = ActivationFunction.get("tanh")
        np.testing.assert_array_almost_equal(tanh_func(x), np.tanh(x))

        # ReLU
        relu_func = ActivationFunction.get("relu")
        np.testing.assert_array_equal(relu_func(x), np.array([0, 0, 1]))

        # Sigmoid
        sigmoid_func = ActivationFunction.get("sigmoid")
        expected_sigmoid = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid_func(x), expected_sigmoid)

        # Test invalid name
        with self.assertRaises(ValueError):
            ActivationFunction.get("invalid_activation")

if __name__ == "__main__":
    unittest.main()
