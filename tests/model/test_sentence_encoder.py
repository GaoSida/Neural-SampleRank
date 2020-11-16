"""Test the sentence encoders.
"""

import pytest
import torch

from nsr.model.sentence_encoder import BiRNN


def test_invalid_rnn_type():
    with pytest.raises(AssertionError):
        rnn = BiRNN(10, 10, 2, 0.5, "random_cell")


def test_bilstm():
    bi_lstm = BiRNN(10, 15, 2, 0.5, "LSTM")
    dummy_input = torch.rand(3, 7, 10)
    dummy_lengths = torch.tensor([7, 5, 2])
    
    outputs = bi_lstm(dummy_input, dummy_lengths)
    assert outputs.shape == (3, 7, 30)
    
    # Check padding
    zero_state = torch.zeros(30)
    assert not (outputs[0, 6, :] == zero_state).all()
    assert (outputs[1, 5, :] == zero_state).all()
    assert (outputs[1, 6, :] == zero_state).all()
    assert (outputs[2, 2, :] == zero_state).all()
    assert (outputs[2, 4, :] == zero_state).all()
    