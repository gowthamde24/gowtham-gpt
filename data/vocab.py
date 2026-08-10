"""Character-level vocabulary: the tokenizer the training/generation pipeline uses."""

from typing import Dict, List


class CharVocabulary:
    """Maps between characters and integer token IDs for a fixed piece of text.

    Character-level tokenization keeps the pipeline simple and dependency-free — no
    merge rules to train (see `data/tokenizer.py` for a from-scratch BPE reference) —
    at the cost of longer sequences per unit of text.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)
