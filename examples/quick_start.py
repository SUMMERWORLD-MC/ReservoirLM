import numpy as np

from reservoir_lm.data_prep.tokenizer import Tokenizer
from reservoir_lm.data_prep.encoder import TimeSeriesEncoder
from reservoir_lm.reservoir_core.cell import ReservoirCell
from reservoir_lm.reservoir_core.simulator import ReservoirSimulator
from reservoir_lm.readout_layer.layer import ReadoutLayer
from reservoir_lm.readout_layer.trainer import Trainer
from reservoir_lm.readout_layer.predictor import Predictor

def main():
    """A quick demonstration of the ReservoirLM library."""
    # 1. Data Preparation
    print("Step 1: Preparing the data...")
    texts = [
        "reservoir computing is a framework for computation",
        "it is derived from recurrent neural network models",
        "the reservoir is a fixed random recurrent neural network",
        "only the readout layer is trained"
    ]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    vocab_size = tokenizer.vocab_size
    embedding_dim = 50
    encoder = TimeSeriesEncoder(vocab_size, embedding_dim)
    encoded_sequences = encoder.encode(sequences)

    # We'll use the first sequence for this example
    input_sequence = encoded_sequences[0]
    target_ids = sequences[0][1:]

    print(f"Vocabulary size: {vocab_size}")
    print(f"Input sequence shape: {input_sequence.shape}")
    print("-" * 30)

    # 2. Reservoir Simulation
    print("Step 2: Simulating the reservoir...")
    n_reservoir = 200
    reservoir_cell = ReservoirCell(
        n_reservoir=n_reservoir,
        n_input=embedding_dim,
        spectral_radius=0.99,
        connectivity=0.1
    )
    simulator = ReservoirSimulator(reservoir_cell)

    # Simulate and get the reservoir states. For training, we keep all states.
    reservoir_states = simulator.simulate(input_sequence, warmup_steps=0)

    print(f"Reservoir states shape: {reservoir_states.shape}")
    print("-" * 30)

    # 3. Readout Training
    print("Step 3: Training the readout layer...")

    # The reservoir states are the input (X) for the readout layer.
    # We use the state at time t to predict the token at t+1.
    X = reservoir_states[:-1]

    # The target (Y) is the one-hot encoded version of the next token.
    Y = np.eye(vocab_size)[target_ids]

    if X.shape[0] != Y.shape[0]:
        raise ValueError("Mismatch between number of states and targets.")

    readout_layer = ReadoutLayer(n_reservoir=n_reservoir, output_dim=vocab_size)
    trainer = Trainer(readout_layer, method="ridge", alpha=1e-4)
    trainer.train(X, Y)

    print("Readout layer trained successfully.")
    print(f"W_out shape: {readout_layer.W_out.shape}")
    print("-" * 30)

    # 4. Prediction
    print("Step 4: Making a prediction...")
    predictor = Predictor(readout_layer)

    # Use the last reservoir state to predict the next token
    last_state = reservoir_states[-1].reshape(1, 1, -1)
    probabilities = predictor.predict(last_state)

    predicted_token_id = np.argmax(probabilities)
    predicted_word = tokenizer.index_word.get(predicted_token_id, "unknown")

    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Predicted next token ID: {predicted_token_id}")
    print(f"Predicted next word: '{predicted_word}'")
    print("-" * 30)

if __name__ == "__main__":
    main()
