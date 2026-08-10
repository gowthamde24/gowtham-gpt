from typing import List, Dict


class GreedyTokenizer:
    """Greedy longest-match tokenizer over a fixed vocabulary, with helpers for analyzing tokenization efficiency."""

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        """Tokenizes each number's string form, returning the token list for each."""
        result = []
        for num in numbers:
            text = str(num)
            tokens = self._greedy_tokenize(text, vocab)
            result.append(tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        """Number of tokens `text` decomposes into under greedy tokenization."""
        tokens = self._greedy_tokenize(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        """Tokens-per-word ratio; higher means more expensive, less efficient tokenization."""
        tokens = self._greedy_tokenize(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)

    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            best = None
            for length in range(len(text) - i, 0, -1):
                substr = text[i:i + length]
                if substr in vocab:
                    best = substr
                    break
            if best is None:
                tokens.append(text[i])
                i += 1
            else:
                tokens.append(best)
                i += len(best)
        return tokens
