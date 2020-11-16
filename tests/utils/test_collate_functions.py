import os

import torch
from torch.utils.data import DataLoader

from nsr.utils.conll_dataset import ConllDataset
from nsr.utils.collate_functions import generate_sentence_batch
from nsr.utils.collate_functions import generate_document_batch
from nsr.graph.crf_builder import SeqTagCRFBuilder


def test_conll_dataloader(shared_datadir, dummy_vocabs):
    """Test the CoNLL dataset loader with batching and padding.
    """
    token_vocab, char_vocab, label_vocab = dummy_vocabs
    # Check on the <pad> and <unk> index
    assert char_vocab.stoi["<pad>"] == 1
    assert char_vocab.stoi["<unk>"] == 0
    dataset = ConllDataset(os.path.join(shared_datadir, "conll_sample.txt"),
                           token_vocab, char_vocab, label_vocab)
    dataset = DataLoader(dataset, batch_size=3,
                         collate_fn=generate_sentence_batch)

    # Check the one batch
    for batch in dataset:
        tokens, token_chars, lengths, labels = batch
        assert lengths.tolist() == [5, 3, 1]
        assert tokens.shape == (3, 5)
        assert token_chars.shape == (15, 7)  # max char len is 7
        assert labels.shape == (3, 5)
        
        assert (tokens == torch.tensor([
            [5, 6, 7, 2, 0], [6, 4, 3, 1, 1], [0, 1, 1, 1, 1]
        ])).all()
        assert (labels == torch.tensor([
            [3, 5, 6, 6, 2], [3, 6, 6, 1, 1], [3, 1, 1, 1, 1]
        ])).all()
        
        # Check padding of characters
        assert token_chars[0].tolist() == [17, 32, 47, 32, 45, 1, 1]
        assert token_chars[3].tolist() == [28, 47, 1, 1, 1, 1, 1]
        assert token_chars[7].tolist() == [35, 28, 43, 43, 52, 1, 1]
        assert (token_chars[1] == token_chars[5]).all()  # "Such"
        assert token_chars[8].tolist() == token_chars[9].tolist() == [1] * 7
        assert token_chars[10].tolist() == [3, 32, 52, 42, 41, 30, 0]


def test_conll_doc_dataloader(shared_datadir, dummy_vocabs):
    """Test the CoNLL dataset loader for documents, with batching, padding and
    re-aligning on the graph dependencies.
    """
    token_vocab, char_vocab, label_vocab = dummy_vocabs
    dataset = ConllDataset(
        os.path.join(shared_datadir, "conll_doc_sample.txt"),
        token_vocab, char_vocab, label_vocab, doc_mode=True,
        graph_builder=SeqTagCRFBuilder(skip_chain_enabled=True))

    dataset = DataLoader(dataset, batch_size=3,
                         collate_fn=generate_document_batch)
    
    # Check the one batch
    for batch in dataset:
        tokens, token_chars, sentence_lengths, \
            factor_dependencies, strings, labels = batch
        
        assert sentence_lengths.tolist() == [1, 5, 3, 4, 1, 3, 1, 5, 2]
        assert tokens.shape == (9, 5)
        assert token_chars.shape == (45, 10)
        assert labels.shape == (9, 5)
        
        # Check values
        assert (tokens[0:3] == torch.tensor([
            [0, 1, 1, 1, 1], [5, 6, 7, 2, 0], [6, 4, 3, 1, 1]
        ])).all()
        assert (labels[5:] == torch.tensor([
            [3, 6, 6, 1, 1], [6, 1, 1, 1, 1], [2, 4, 4, 6, 6], [3, 5, 1, 1, 1]
        ])).all()
        
        # Check the factor dependencies
        assert len(factor_dependencies) == 3
        assert factor_dependencies["unary"] == [
            (0,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,),
            (15,), (16,), (17,), (18,), (20,), (25,), (26,), (27,), (30,),
            (35,), (36,), (37,), (38,), (39,), (40,), (41,)
        ]
        
        assert factor_dependencies["transition"] == [
            (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12),
            (15, 16), (16, 17), (17, 18), (25, 26), (26, 27),
            (35, 36), (36, 37), (37, 38), (38, 39), (40, 41)
        ]
        
        assert factor_dependencies["skip"] == [
            (5, 15), (6, 10), (35, 41)
        ]
        
        # Check the string values
        assert len(strings) == 9
        assert strings == [
            "-DOCSTART-",
            "Peter Such won at London",
            "Such is happy",
            "Peter returned home today",
            "-DOCSTART-",
            "Beyoncé releases album",
            "-DOCSTART-",
            "Nook 's Cranny opens today",
            "Tom Nook"
        ]
