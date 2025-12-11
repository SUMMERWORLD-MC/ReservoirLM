import numpy as np
from typing import Callable, List
from .cell import ReservoirCell
from .activation import ActivationFunction


class ReservoirSimulator:
    """
    Computes the reservoir's state trajectory based on an input time series.
    """

    def __init__(self, reservoir_cell: ReservoirCell, activation: str = "tanh"):
        """
        Args:
            reservoir_cell (ReservoirCell): The reservoir cell with initialized weights.
            activation (str): The name of the activation function to use ('tanh', 'relu', 'sigmoid').
        """
        self.cell = reservoir_cell
        self.activation_func = ActivationFunction.get(activation)

    def simulate(
        self, u_t: np.ndarray, warmup_steps: int = 0
    ) -> np.ndarray:
        """
        Runs the simulation and returns the reservoir's state trajectory.

        Args:
            u_t (np.ndarray): The input time series with shape (sequence_length, n_input).
            warmup_steps (int): The number of initial steps to discard.

        Returns:
            np.ndarray: The reservoir's state trajectory with shape (sequence_length - warmup_steps, n_reservoir).
        """
        sequence_length, _ = u_t.shape
        states = np.zeros((sequence_length, self.cell.n_reservoir))
        x_t = np.zeros(self.cell.n_reservoir)

        for t in range(sequence_length):
            # State update equation: x(t+1) = f(W_in @ u(t) + W @ x(t) + W_bias)
            state_update = self.cell.W_in @ u_t[t] + self.cell.W @ x_t + self.cell.W_bias
            x_t = self.activation_func(state_update)
            states[t] = x_t

        # Discard warmup steps
        return states[warmup_steps:]

    def simulate_batch(
        self, u_batch: np.ndarray, warmup_steps: int = 0
    ) -> np.ndarray:
        """
        Runs the simulation for a batch of input sequences.

        Args:
            u_batch (np.ndarray): The input batch with shape (batch_size, sequence_length, n_input).
            warmup_steps (int): The number of initial steps to discard.

        Returns:
            np.ndarray: The reservoir's state trajectories for the batch, with shape
                        (batch_size, sequence_length - warmup_steps, n_reservoir).
        """
        batch_size = u_batch.shape[0]
        results = []

        for i in range(batch_size):
            states = self.simulate(u_batch[i], warmup_steps=warmup_steps)
            results.append(states)

        return np.array(results)
