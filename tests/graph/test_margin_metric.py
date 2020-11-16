"""Test margin metrics.
"""
from typing import List

from nsr.graph.margin_metric import NegHammingDistance
from nsr.graph.factor_graph import LabelNode


def set_label_node_values(label_nodes: List[LabelNode], label_vals: List[int]):
    for node, val in zip(label_nodes, label_vals):
        node.current_value = val


def test_neg_hamming_loss():
    ground_truth = [2, 0, 4, 3, 2, 3, 0, 2, 1, 1]
    metric = NegHammingDistance(ground_truth)
    current_sample = [LabelNode(i, 5) for i in range(10)]
    
    label_vals = [0] * 10
    set_label_node_values(current_sample, label_vals)
    assert metric.compute_metric(label_vals) == -6
    
    current_sample[0].current_value = 2
    assert metric.incremental_compute_metric(
        current_sample, -6, label_vals, [0]) == -5
    label_vals[0] = 2
    
    current_sample[3].current_value = 3
    assert metric.incremental_compute_metric(
        current_sample, -5, label_vals, [3]) == -4
    label_vals[3] = 3
    
    current_sample[5].current_value = 3
    current_sample[7].current_value = 2
    assert metric.incremental_compute_metric(
        current_sample, -4, label_vals, [5, 7]) == -2
    label_vals[5] = 3
    label_vals[7] = 2
    
    current_sample[5].current_value = 4
    current_sample[7].current_value = 2
    assert metric.incremental_compute_metric(
        current_sample, -2, label_vals, [5, 7]) == -3
    label_vals[5] = 4
    label_vals[7] = 2
    
    current_sample[0].current_value = 3
    assert metric.incremental_compute_metric(
        current_sample, -3, label_vals, [0]) == -4
    label_vals[0] = 3
