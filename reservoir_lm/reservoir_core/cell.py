import numpy as np
from scipy.sparse import random as sparse_random, spmatrix
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from typing import Union


class ReservoirCell:
    """
    Defines the structure and parameters of the reservoir, including random weight matrices.
    """

    def __init__(
        self,
        n_reservoir: int,
        n_input: int,
        spectral_radius: float = 0.99,
        connectivity: float = 0.1,
        input_scaling: float = 1.0,
        bias_scaling: float = 1.0,
    ):
        """
        Args:
            n_reservoir (int): The number of reservoir units (Nr).
            n_input (int): The dimensionality of the input features.
            spectral_radius (float): The spectral radius of the recurrent weight matrix.
            connectivity (float): The density of the recurrent weight matrix.
            input_scaling (float): The scaling factor for the input weights.
            bias_scaling (float): The scaling factor for the bias weights.
        """
        self.n_reservoir = n_reservoir
        self.n_input = n_input
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        self.W_in = self._initialize_input_weights()
        self.W = self._initialize_recurrent_weights()
        self.W_bias = self._initialize_bias()

    def _initialize_input_weights(self) -> np.ndarray:
        """Initializes the input weight matrix W_in."""
        return (np.random.rand(self.n_reservoir, self.n_input) - 0.5) * self.input_scaling

    def _initialize_recurrent_weights(self) -> Union[np.ndarray, spmatrix]:
        """Initializes the sparse recurrent weight matrix W."""
        # Generate a sparse random matrix
        W = sparse_random(
            self.n_reservoir,
            self.n_reservoir,
            density=self.connectivity,
            data_rvs=lambda size: np.random.randn(size),
        )

        # Normalize the spectral radius using scipy's sparse eigensolver
        try:
            eigenvalues = eigs(W, k=1, which="LM", return_eigenvectors=False)
            current_spectral_radius = np.max(np.abs(eigenvalues))
            if current_spectral_radius > 0:
                W = W * (self.spectral_radius / current_spectral_radius)
        except (np.linalg.LinAlgError, ArpackNoConvergence):
            # Fallback to dense matrix if sparse solver fails
            W_dense = W.toarray()
            eigenvalues, _ = np.linalg.eig(W_dense)
            current_spectral_radius = np.max(np.abs(eigenvalues))
            if current_spectral_radius > 0:
                W = W_dense * (self.spectral_radius / current_spectral_radius)

        return W

    def _initialize_bias(self) -> np.ndarray:
        """Initializes the bias vector W_bias."""
        return (np.random.rand(self.n_reservoir) - 0.5) * self.bias_scaling
