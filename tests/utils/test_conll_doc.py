"""Test doc version of CoNLL data loader.
"""
import os

from nsr.utils.conll_dataset import ConllDataset
from nsr.graph.crf_builder import SeqTagCRFBuilder


def check_doc_content(dataset):
    assert len(dataset) == 3

    doc = dataset[0]
    assert len(doc["tokens"]) == len(doc["token_chars"]) \
        == len(doc["strings"]) == len(doc["labels"]) == 4
    assert doc["tokens"][0].tolist() == [0]
    assert doc["tokens"][1].tolist() == [5, 6, 7, 2, 0]
    assert len(doc["token_chars"][1]) == 5
    assert doc["token_chars"][1][0].tolist() == [17, 32, 47, 32, 45]  # "Peter"
    assert doc["token_chars"][2][2].tolist() == [35, 28, 43, 43, 52]  # "happy"
    assert doc["token_chars"][2][0].tolist() == [20, 48, 30, 35]  # "Such"
    assert doc["token_chars"][1][1].tolist() == [20, 48, 30, 35]  # "Such"
    assert doc["tokens"][2].tolist() == [6, 4, 3]
    assert len(doc["token_chars"][2]) == 3
    assert doc["labels"][1].tolist() == [3, 5, 6, 6, 2]
    assert doc["labels"][2].tolist() == [3, 6, 6]
    assert doc["labels"][3].tolist() == [3, 6, 6, 6]
    assert doc["strings"] == [['-DOCSTART-'],
                              ['Peter', 'Such', 'won', 'at', 'London'],
                              ['Such', 'is', 'happy'],
                              ['Peter', 'returned', 'home', 'today']]
    
    doc = dataset[1]
    assert len(doc["tokens"]) == len(doc["token_chars"]) \
        == len(doc["strings"]) == len(doc["labels"]) == 2
    assert doc["strings"] == [['-DOCSTART-'], ['Beyoncé', 'releases', 'album']]
    
    doc = dataset[2]
    assert len(doc["tokens"]) == len(doc["token_chars"]) \
        == len(doc["strings"]) == len(doc["labels"]) == 3
    assert doc["labels"][1].tolist() == [2, 4, 4, 6, 6]
    assert doc["strings"] == [['-DOCSTART-'],
                              ['Nook', "'s", 'Cranny', 'opens', 'today'],
                              ['Tom', 'Nook']]


def test_conll_doc_dataset(shared_datadir, dummy_vocabs):
    token_vocab, char_vocab, label_vocab = dummy_vocabs
    dataset = ConllDataset(
        os.path.join(shared_datadir, "conll_doc_sample.txt"),
        token_vocab, char_vocab, label_vocab, doc_mode=True)

    for doc in dataset:
        assert set(doc.keys()) == set(["tokens", "token_chars",
                                       "strings", "labels"])
    check_doc_content(dataset)


def test_conll_doc_dataset_with_graph_builder(shared_datadir, dummy_vocabs):
    token_vocab, char_vocab, label_vocab = dummy_vocabs
    dataset = ConllDataset(
        os.path.join(shared_datadir, "conll_doc_sample.txt"),
        token_vocab, char_vocab, label_vocab, doc_mode=True,
        graph_builder=SeqTagCRFBuilder(skip_chain_enabled=True))
    
    for doc in dataset:
        assert set(doc.keys()) == set(["tokens", "token_chars",
                                       "strings", "labels", "graph"])
    
    check_doc_content(dataset)

    doc = dataset[0]
    assert len(doc["graph"]["unary"]) == 13
    assert len(doc["graph"]["transition"]) == 9
    assert doc["graph"]["skip"] == [((1, 0), (3, 0)), ((1, 1), (2, 0))]
    
    doc = dataset[1]
    assert len(doc["graph"]["unary"]) == 4
    assert len(doc["graph"]["transition"]) == 2
    assert len(doc["graph"]["skip"]) == 0
    
    doc = dataset[2]
    assert len(doc["graph"]["unary"]) == 8
    assert len(doc["graph"]["transition"]) == 5
    assert doc["graph"]["skip"] == [((1, 0), (2, 1))]
