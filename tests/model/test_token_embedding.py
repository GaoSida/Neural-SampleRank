"""Test the token embedding layer.
"""
import os
from collections import Counter

import torch
from torchtext.vocab import Vocab, GloVe, FastText

from nsr.model.token_embedding import CharCNN, TokenEmbedding


def test_char_cnn():
    char_cnn = CharCNN(vocab_size=5, embed_dim=10, kernel_size=3,
                       num_kernels=20, dropout=0.5)
    dummy_token_chars = torch.tensor([
        [4, 2, 0, 3, 1, 1, 1],
        [3, 4, 2, 2, 0, 1, 1]
    ])
    
    char_cnn.train()  # Turn on dropout
    output_1 = char_cnn(dummy_token_chars)
    assert output_1.shape == (2, 20)
    output_2 = char_cnn(dummy_token_chars)
    assert output_2.shape == (2, 20)
    assert not (output_2 == output_1).all()  # due to random dropout
    
    char_cnn.eval()  # Turn off dropout
    output_1 = char_cnn(dummy_token_chars)
    assert output_1.shape == (2, 20)
    output_2 = char_cnn(dummy_token_chars)
    assert output_2.shape == (2, 20)
    assert (output_2 == output_1).all()
    
    # Check the unknown and padding embeddings
    assert (char_cnn.embeddings(torch.tensor([1])) 
            == torch.zeros([10, ])).all()  # Padding embedding is all zeros
    assert not (char_cnn.embeddings(torch.tensor([0]))
                == torch.zeros([10, ])).all()  # <unk>, not all zeros


def test_token_embedding(shared_datadir):
    """Test the initialization and concatenation of pretrained embedding.
    """
    # Cache file is ignored by git
    glove = GloVe(name="6B", dim=100,
                  cache=os.path.join(shared_datadir, "cache"),
                  unk_init=torch.Tensor.normal_)
    # Wrap glove around as a Vocab with <unk> and <pad>
    vocab = Vocab(Counter(list(glove.stoi.keys())), vectors=glove)
    assert len(vocab.stoi) == 400000 + 2  # glove + <unk>, <pad>
    
    # Check that unk token is not all zero
    assert not (vocab.vectors[vocab.stoi["<unk>"]] == torch.zeros(100)).all()
    # Check if some tokens are correctly lookedup
    assert torch.max(torch.abs(
        vocab.vectors[vocab.stoi["the"]][:6] - torch.tensor(
            [-0.038194, -0.24487, 0.72812, -0.39961, 0.083172, 0.043953]
        ))) < 1e-6  # Values looked up from txt file
    
    # Check the actual token embedding layer
    char_cnn = CharCNN(vocab_size=5, embed_dim=10, kernel_size=3,
                       num_kernels=20, dropout=0.5)
    token_embed = TokenEmbedding(vocab.vectors, char_cnn)
    embeddings = token_embed(
        tokens=torch.tensor([
            [vocab.stoi["the"], vocab.stoi["melancholy"], vocab.stoi["<unk>"]]
        ]),  # Batch size = 1
        token_chars=torch.tensor(
            [[2, 0, 4, 1], [3, 2, 1, 1], [4, 1, 1, 1]]  # dummy values
        )
    )
    
    assert embeddings.shape == (1, 3, 120)
    # Check the value of pretrained embedding
    assert (embeddings[0, 0, :100] == vocab.vectors[vocab.stoi["the"]]).all()
    assert (embeddings[0, 2, :100] == vocab.vectors[vocab.stoi["<unk>"]]).all()
    assert torch.max(torch.abs(
        embeddings[0, 1, :6] - torch.tensor(
            [-0.12609, 0.28185, 0.99061, 0.11279, -0.063109, 1.1661]
        ))) < 1e-6


def test_de_nl_embedding(shared_datadir):
    """Test the PyTorch API for fasttext embeddings for de and nl
    """
    fasttext_de = FastText(language="de",
                           cache=os.path.join(shared_datadir, "cache"),
                           unk_init=torch.Tensor.normal_)
    vocab = Vocab(Counter(list(fasttext_de.stoi.keys())), vectors=fasttext_de)
    
    # Check that unk token is not all zero
    assert not (vocab.vectors[vocab.stoi["<unk>"]] == torch.zeros(300)).all()
    # Check if some tokens are correctly lookedup
    assert torch.max(torch.abs(
        vocab.vectors[vocab.stoi["Großer".lower()]][:10] - torch.tensor(
            [-0.4225,  0.1097, -0.1615,  0.1069, -0.0628,  0.0133, -0.1279, 
             -0.0307, 0.2388,  0.0212]
        ))) < 1e-4
    
    assert torch.max(torch.abs(
        vocab.vectors[vocab.stoi["dem".lower()]][:10] - torch.tensor(
            [-0.2589, -0.0089, -0.5246, -0.3139,  0.1133,  0.1864, -0.0811,
             -0.0726, 0.0264,  0.0278]
        ))) < 1e-4
    
    assert torch.max(torch.abs(
        vocab.vectors[vocab.stoi["und".lower()]][:10] - torch.tensor(
            [-0.1778, -0.0642, -0.0947, -0.1643, -0.0839,  0.0527, -0.1037,
             0.0792, -0.0489, -0.0030]
        ))) < 1e-4
