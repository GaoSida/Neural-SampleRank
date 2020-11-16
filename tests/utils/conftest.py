import os
import pytest
from collections import Counter

from torchtext.vocab import Vocab


@pytest.fixture()
def dummy_vocabs():
    token_list = ["at", "happy", "is", "peter", "such", "won"]
    char_list = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    char_list += [chr(c) for c in range(ord('a'), ord('z') + 1)]
    label_list = ["B-LOC", "B-PER", "I-LOC", "I-PER", "O"]
    # torchtext vocabs are sorted by dictionary order internally
    return Vocab(Counter(token_list)), Vocab(Counter(char_list)), \
        Vocab(Counter(label_list))
