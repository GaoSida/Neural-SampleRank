"""Tests for the unary factor.
"""
import torch

from nsr.model.unary_factor import UnaryFactor


def test_unary_factor():
    factor = UnaryFactor(20, 11, 15, 0.5)
    dummy_input = torch.rand(3, 7, 20)
    
    scores = factor(dummy_input)
    assert scores.shape == (3, 7, 11)


def test_unary_factor_flat_batch():
    factor = UnaryFactor(20, 11, 15, 0.5)
    dummy_input = torch.rand(21, 20)
    
    scores = factor(dummy_input)
    assert scores.shape == (21, 11)
