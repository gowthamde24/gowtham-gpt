"""Data pipeline: production components plus reference tokenization exercises.

Production path (what `train.py` / `generate.py` use):
    CharVocabulary — build a char-level vocab, encode/decode text
    get_batch      — sample random (X, Y) training windows

Reference implementations, not used by the main pipeline:
    learn_bpe_merges        — from-scratch byte-pair-encoding trainer
    GreedyTokenizer         — greedy longest-match tokenization over a fixed vocab
    build_sentiment_dataset — word-level batching for a toy sentiment task
    build_word_batches      — word-level batch sampling (see get_batch for the
                               character-level version the pipeline actually uses)
"""

from .vocab import CharVocabulary
from .loader import get_batch
from .tokenizer import learn_bpe_merges
from .tokenizer_utils import GreedyTokenizer
from .nlp_preprocessing import build_sentiment_dataset
from .dataset import build_word_batches

__all__ = [
    "CharVocabulary",
    "get_batch",
    "learn_bpe_merges",
    "GreedyTokenizer",
    "build_sentiment_dataset",
    "build_word_batches",
]
