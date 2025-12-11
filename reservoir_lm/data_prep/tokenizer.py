import json
from collections import Counter
from typing import List, Dict, Union


class Tokenizer:
    """Manages text tokenization and vocabulary conversion."""

    def __init__(self, unk_token: str = "<unk>", pad_token: str = "<pad>"):
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self._init_vocab()

    def _init_vocab(self):
        """Initializes the vocabulary with special tokens."""
        self.word_index = {self.pad_token: 0, self.unk_token: 1}
        self.index_word = {0: self.pad_token, 1: self.unk_token}

    def fit_on_texts(self, texts: List[str]):
        """Creates a vocabulary from a list of texts."""
        words = " ".join(texts).split()
        word_counts = Counter(words)
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

        for word in sorted_words:
            if word not in self.word_index:
                index = len(self.word_index)
                self.word_index[word] = index
                self.index_word[index] = word

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Converts texts to sequences of token IDs."""
        sequences = []
        for text in texts:
            sequence = [
                self.word_index.get(word, self.word_index[self.unk_token])
                for word in text.split()
            ]
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """Converts sequences of token IDs back to texts."""
        texts = []
        for sequence in sequences:
            text = " ".join(
                [self.index_word.get(idx, self.unk_token) for idx in sequence]
            )
            texts.append(text)
        return texts

    def save_vocab(self, path: str):
        """Saves the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word_index": self.word_index}, f, ensure_ascii=False, indent=4)

    def load_vocab(self, path: str):
        """Loads the vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word_index = data["word_index"]
        self.index_word = {idx: word for word, idx in self.word_index.items()}

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.word_index)
