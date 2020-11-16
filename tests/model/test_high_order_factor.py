"""Tests for the binary scoring factor.
"""
import torch

from nsr.model.high_order_factor import HighOrderFactor


def test_binary_fcator():
    factor = HighOrderFactor(input_dim=15, order=2, ops=[
        lambda x: torch.sum(x, dim=0), lambda x: torch.max(x, dim=0)[0]],
        output_dim=20, hidden_dim=50, dropout=0.5)
    
    states_a = torch.rand(7, 15)
    states_b = torch.rand(7, 15)
    
    scores = factor([states_a, states_b])
    assert scores.shape == (7, 20)


def test_tertiary_fcator():
    factor = HighOrderFactor(input_dim=15, order=3, ops=[
        lambda x: torch.sum(x, dim=0), lambda x: torch.max(x, dim=0)[0]],
        output_dim=20, hidden_dim=50, dropout=0.5)
    
    states_a = torch.rand(7, 15)
    states_b = torch.rand(7, 15)
    states_c = torch.rand(7, 15)
    
    scores = factor([states_a, states_b, states_c])
    assert scores.shape == (7, 20)
