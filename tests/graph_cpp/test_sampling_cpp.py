"""Test the cpp code with Python binding with PyTest.
Building PyTorch C++ code is difficult so we reuse what's available for the
extension and binding.
"""
import random

import torch
import numpy as np

import cpp_tests


def test_unary_node_cpp():
    results = cpp_tests.test_unary_node()
    assert results[0:4] == [0, 1, 3, 7]
    assert abs(results[4] - 101.7) < 1e-4


def test_binary_node_cpp():
    results = cpp_tests.test_binary_node()
    assert results == [37.0, 37.0, 25.0, 62.0]
    

def test_graph_adjacency_list():
    results = cpp_tests.test_graph_adjacency_list()
    # Label nodes
    assert results[0] == 5.0
    assert results[1:16] == [0, 3, 2, 1, 3, 3, 2, 3, 3, 3, 3, 3, 4, 3, 3]
    # Factor nodes
    assert results[16] == 3
    assert results[17] == 5  # Unary nodes
    assert results[18:33] == [1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 4, 1]
    assert results[33] == 3  # Binary nodes
    assert results[34:46] == [1, 1, 4, 1, 2, 1, 3, 1, 3, 1, 4, 1]
    assert results[46] == 1  # Tertiary nodes
    assert results[47:53] == [0, 1, 1, 1, 2, 1]
    assert len(results) == 53


def test_factor_graph_compute_scores():
    results = cpp_tests.test_factor_graph_compute_scores()
    
    assert results[0:2] == [31.0, 31.0]
    assert results[2:5] == [0.0, 31.0, 30.0]
    assert results[5] == 31.0
    
    assert len(results) == 6


def test_init_with_oracle():
    ground_truth = [random.randint(0, 9) for _ in range(100)]
    
    label_vals = cpp_tests.test_init_with_oracle(ground_truth, -1, True)
    assert sum([p == t for p, t in zip(label_vals, ground_truth)]) < 30
    
    label_vals = cpp_tests.test_init_with_oracle(ground_truth, 1, False)
    assert sum([p == t for p, t in zip(label_vals, ground_truth)]) == 100
    
    label_vals = cpp_tests.test_init_with_oracle(ground_truth, 0.5, False)
    assert sum([p == t for p, t in zip(label_vals, ground_truth)]) > 30
    assert sum([p == t for p, t in zip(label_vals, ground_truth)]) < 70


def test_unary_node_conditional_scores():
    results = cpp_tests.test_unary_node_conditional_scores()
    
    conditional_score = results[0]
    assert conditional_score.tolist() == list(range(9))
    
    conditional_score = results[1]
    assert conditional_score.shape == (1, 9, 1)
    assert conditional_score.flatten().tolist() == list(range(9))
    
    conditional_score = results[2]
    assert conditional_score.shape == (1, 1)
    assert conditional_score.item() == 3


def test_binary_node_conditional_scores():
    results = cpp_tests.test_binary_node_conditional_scores()
    
    conditional_score = results[0]
    assert conditional_score.shape == (9, )
    assert conditional_score.tolist() == [2, 9, 16, 23, 30, 37, 44, 51, 58]
    
    conditional_score = results[1]
    assert conditional_score.shape == (7, )
    assert conditional_score.tolist() == [35, 36, 37, 38, 39, 40, 41]
    
    score_table = torch.tensor(range(63)).view(9, 7)
    conditional_score = results[2]
    assert conditional_score.shape == (9, 7)
    assert (conditional_score == score_table).all()
    
    conditional_score = results[3]
    assert conditional_score.shape == (7, 9)
    assert (conditional_score == score_table.t()).all()
    
    conditional_score = results[4]
    assert conditional_score.shape == (1, 9, 1, 7)
    assert (conditional_score.view(9, 7) == score_table).all()
    # Degenerate case
    conditional_score = results[5]
    assert conditional_score.shape == (1, 1, 1)
    assert conditional_score.item() == 37


def test_gibbs_sample_probabilities():
    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [0], 1.0)
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([2, 12, 22])).all()
    
    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [0], 0.1)
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([20, 120, 220])).all()

    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [1], 1.0)
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([13, 20, 27])).all()

    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [3], 1.0)
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([8, 13, 18])).all()

    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [1, 2], 1.0)
    assert conditional_scores.shape == (3, 3)
    assert (conditional_scores == torch.tensor([
        [12, 17, 22], [19, 24, 29], [26, 31, 36]
    ])).all()
    
    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [2, 1], 1.0)
    assert conditional_scores.shape == (3, 3)
    assert (conditional_scores == torch.tensor([
        [12, 19, 26], [17, 24, 31], [22, 29, 36]
    ])).all()
    
    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [3, 4], 1.0)
    assert conditional_scores.shape == (3, 3)
    assert (conditional_scores == torch.tensor([
        [6, 9, 12], [11, 14, 17], [16, 19, 22]
    ])).all()
    
    conditional_scores = cpp_tests.test_gibbs_sample_probabilities(
        [1, 0, 2, 1, 2], [0, 1, 2], 1.0)
    assert conditional_scores.shape == (3, 3, 3)
    assert (conditional_scores == torch.tensor([
        [[3, 8, 13], [10, 15, 20], [17, 22, 27]],
        [[13, 18, 23], [20, 25, 30], [27, 32, 37]],
        [[23, 28, 33], [30, 35, 40], [37, 42, 47]]
    ])).all()


def test_gibbs_sample_forced():
    results = cpp_tests.test_gibbs_sample_forced()
    assert results[0] == [2, 1, 0, 1, 2]
    assert results[1] == [2, 1, 0, 2, 1]
    assert results[2] == [2, 1, 0, 2, 2]
    assert results[3] == [2, 1, 1, 2, 2]
    assert results[4] == [2, 1, 1, 2, 2]
    assert len(results) == 5


def test_neg_hamming_loss():
    results = cpp_tests.test_neg_hamming_loss()
    # assert results == [-6, -5, -4, -2, -3, -4]

    assert results[0] == -6
    assert set(results[1:7]) == set([0, 2, 3, 4, 5, 7])
    assert results[7] == -5
    assert set(results[8:13]) == set([2, 3, 4, 5, 7])
    assert results[13] == -4
    assert set(results[14:18]) == set([2, 4, 5, 7])
    assert results[18] == -2
    assert set(results[19:21]) == set([2, 4])
    assert results[21] == -3
    assert set(results[22:25]) == set([2, 4, 5])
    assert results[25] == -4
    assert set(results[26:30]) == set([0, 2, 4, 5])
    assert len(results) == 30
