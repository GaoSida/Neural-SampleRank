"""
Test the CoNLL data loader.
"""
import os

import torch
from torch.utils.data import DataLoader

from nsr.utils.conll_dataset import count_conll_vocab, compute_f1, ConllDataset


def test_count_conll_vocab(shared_datadir):
    char_counter, label_counter = count_conll_vocab(
        os.path.join(shared_datadir, "conll_sample.txt"))
    
    assert len(label_counter) == 4
    assert label_counter["O"] == 4
    assert label_counter["B-PER"] == 3
    assert label_counter["I-PER"] == 1
    assert label_counter["B-LOC"] == 1
    
    assert len(char_counter) == 20
    assert char_counter["o"] == 4
    assert char_counter["e"] == 3
    assert char_counter["h"] == 3
    assert char_counter["i"] == 1
    assert char_counter["é"] == 1


def test_conll_dataset(shared_datadir, dummy_vocabs):
    """Test the CoNLL dataset loader for single examples.
    """
    token_vocab, char_vocab, label_vocab = dummy_vocabs
    assert len(token_vocab) == 6 + 2  # plus 2 special tokens
    assert len(char_vocab) == 26 * 2 + 2
    assert len(label_vocab) == 5 + 2
    dataset = ConllDataset(os.path.join(shared_datadir, "conll_sample.txt"),
                           token_vocab, char_vocab, label_vocab)
    assert len(dataset) == 3
    
    # Note: The index order in the Vocab is random due to Counter
    # We only do sanity checks: same string mapped to the same index
    tokens_0, chars_0, labels_0 = \
        dataset[0]["tokens"], dataset[0]["token_chars"], dataset[0]["labels"]
    tokens_1, chars_1, labels_1 = \
        dataset[1]["tokens"], dataset[1]["token_chars"], dataset[1]["labels"]
    tokens_2, chars_2, labels_2 = \
        dataset[2]["tokens"], dataset[2]["token_chars"], dataset[2]["labels"]

    assert len(tokens_0) == len(chars_0) == len(labels_0) == 5
    assert len(tokens_1) == len(chars_1) == len(labels_1) == 3
    assert len(tokens_2) == len(chars_2) == len(labels_2) == 1
 
    assert tokens_0.tolist() == [5, 6, 7, 2, 0]  # "London" is unknown
    assert tokens_1.tolist() == [6, 4, 3]
    assert tokens_2.tolist() == [0]
    
    assert chars_0[0].tolist() == [17, 32, 47, 32, 45]  # "Peter"
    assert chars_1[2].tolist() == [35, 28, 43, 43, 52]  # "happy"
    assert chars_1[0][0] != chars_1[1][1]  # "S" v.s. "s"
    assert chars_2[0][6] == 0  # character "é" is unknown
    # Repeated token "Such"
    assert chars_0[1].tolist() == chars_1[0].tolist() == [20, 48, 30, 35]

    assert labels_0.tolist() == [3, 5, 6, 6, 2]
    assert labels_1.tolist() == [3, 6, 6]
    assert labels_2.tolist() == [3]


def test_write_conll_prediction(shared_datadir, dummy_vocabs):
    token_vocab, char_vocab, label_vocab = dummy_vocabs
    dataset = ConllDataset(os.path.join(shared_datadir, "conll_sample.txt"),
                           token_vocab, char_vocab, label_vocab)
    
    tokens = torch.tensor([
        [5, 6, 7, 2, 0], [6, 4, 3, 1, 1], [0, 1, 1, 1, 1]
    ])
    labels = torch.tensor([
        [3, 5, 6, 6, 2], [3, 6, 6, 1, 1], [3, 1, 1, 1, 1]
    ])
    predictions = torch.tensor([
        [3, 5, 6, 6, 2], [6, 6, 6, 1, 1], [5, 1, 1, 1, 1]
    ])

    assert dataset.write_prediction(tokens, labels, predictions) == \
        "peter B-PER B-PER\n" \
        "such I-PER I-PER\n" \
        "won O O\n" \
        "at O O\n" \
        "<unk> B-LOC B-LOC\n\n" \
        "such B-PER O\n" \
        "is O O\n" \
        "happy O O\n\n" \
        "<unk> B-PER I-PER\n\n"


def test_compute_conll_f1(shared_datadir):
    prediction_dump = \
        "peter B-PER B-PER\n" \
        "such I-PER I-PER\n" \
        "won O O\n" \
        "at O O\n" \
        "<unk> B-LOC B-LOC\n\n" \
        "such B-PER O\n" \
        "is O O\n" \
        "happy O O\n\n" \
        "<unk> B-PER I-PER\n\n"
    # 4 entities: TP 3, FN 1 FP 0
    f1 = compute_f1(prediction_dump, 
                    os.path.join(shared_datadir, "conlleval.pl"))
    assert abs(f1 - 2 * 100 * 75 / (100 + 75)) < 1e-2
