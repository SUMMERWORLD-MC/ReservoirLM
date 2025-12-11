# ReservoirLM (リザバーLM)

ReservoirLM is a Python library for implementing Reservoir Language Models, applying the principles of Reservoir Computing (RC) to text processing and language modeling. It is designed for researchers and developers to easily experiment with, build, and deploy RC-based models.

## Design Philosophy

- **Modularity**: The library is divided into three distinct modules: data processing, reservoir simulation, and readout layer training. This separation allows for independent testing and easy component replacement.
- **Efficiency**: ReservoirLM leverages the strengths of RC by using fast linear regression solvers (like Ridge Regression) for training the readout layer, avoiding computationally expensive gradient descent.
- **Extensibility**: The library is designed to be extensible, allowing users to implement custom reservoir topologies and new readout methods by inheriting from base classes.

## Core Modules

1.  **`data_prep`**: Handles the conversion of raw text into time-series vectors suitable for the reservoir.
    - `Tokenizer`: Manages vocabulary creation and text-to-ID conversion.
    - `TimeSeriesEncoder`: Converts token IDs into dense embedding vectors.
    - `SequenceDataLoader`: Chunks long sequences and creates batches for training.
2.  **`reservoir_core`**: Manages the construction and simulation of the reservoir itself.
    - `ReservoirCell`: Defines the reservoir's structure and initializes its random weight matrices.
    - `ReservoirSimulator`: Runs the time-series simulation, generating the reservoir's state trajectory.
    - `ActivationFunction`: Provides various non-linear activation functions for the reservoir nodes.
3.  **`readout_layer`**: Handles the training of the output layer and making predictions.
    - `ReadoutLayer`: Holds the learnable output weight matrix (`W_out`).
    - `Trainer`: Learns `W_out` using fast linear solvers.
    - `Predictor`: Uses the trained readout layer to compute next-token probabilities.

## Installation

Currently, the library can be used by cloning the repository. The main dependencies are:

```bash
pip install numpy scipy
```

## Quick Start

The following is a basic workflow for building and training a ReservoirLM:

```python
import numpy as np
from reservoir_lm.data_prep.tokenizer import Tokenizer
from reservoir_lm.data_prep.encoder import TimeSeriesEncoder
from reservoir_lm.reservoir_core.cell import ReservoirCell
from reservoir_lm.reservoir_core.simulator import ReservoirSimulator
from reservoir_lm.readout_layer.layer import ReadoutLayer
from reservoir_lm.readout_layer.trainer import Trainer
from reservoir_lm.readout_layer.predictor import Predictor

# 1. Data Preparation
texts = ["reservoir computing is a framework for computation", "it is derived from recurrent neural network models"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

vocab_size = tokenizer.vocab_size
embedding_dim = 50
encoder = TimeSeriesEncoder(vocab_size, embedding_dim)
encoded_sequences = encoder.encode(sequences)
input_sequence = encoded_sequences[0]  # Use the first sequence for this example

# 2. Reservoir Simulation
n_reservoir = 200
reservoir_cell = ReservoirCell(n_reservoir=n_reservoir, n_input=embedding_dim, spectral_radius=0.99)
simulator = ReservoirSimulator(reservoir_cell)
reservoir_states = simulator.simulate(input_sequence)

# 3. Readout Training
# Prepare target data (one-hot encoded next tokens)
target_ids = sequences[0][1:]
targets_one_hot = np.eye(vocab_size)[target_ids]

# The reservoir states corresponding to the targets
X = reservoir_states[:-1]
Y = targets_one_hot

readout_layer = ReadoutLayer(n_reservoir=n_reservoir, output_dim=vocab_size)
trainer = Trainer(readout_layer, method="ridge", alpha=1e-4)
trainer.train(X, Y)

# 4. Prediction
predictor = Predictor(readout_layer)
# Use the last state to predict the next token
last_state = reservoir_states[-1].reshape(1, 1, -1)
probabilities = predictor.predict(last_state)

print("Vocabulary Size:", vocab_size)
print("Predicted probabilities shape:", probabilities.shape)
print("Predicted next token index:", np.argmax(probabilities))
```
