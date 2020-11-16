"""Tests for FactorGraph.
"""
import random

import torch

from nsr.graph.factor_graph import LabelNode, FactorGraph


def test_unary_factor_node(unary_factor_node):
    """Test a factor node when it only touches two label nodes.
    """
    factor_node = unary_factor_node
    assert factor_node.score_table.shape == (9, )
    
    # Check score indexing
    factor_node.label_nodes[0].current_value = 3
    assert factor_node.update_score().item() == 3
    assert factor_node.current_score.item() == 3


def test_binary_factor_node(binary_factor_node):
    """Test a factor node when it touches two label nodes.
    """
    factor_node = binary_factor_node
    assert factor_node.score_table.shape == (9, 7)
    
    factor_node.label_nodes[0].current_value = 5
    factor_node.label_nodes[1].current_value = 2
    assert factor_node.update_score().item() == \
        factor_node.score_table[5, 2] == 37
    assert factor_node.score_table[3, 4] == 25


def test_graph_adjacency_list(five_node_graph):
    """Test if the graph is correctly built.
    """
    graph = five_node_graph
    assert graph.current_score is None
    
    assert len(graph.label_nodes) == 5
    for i, node in enumerate(graph.label_nodes):
        assert node.node_id == i
        assert node.label_space_size == 3
    assert len(graph.label_nodes[0].factor_nodes) == 2
    assert len(graph.label_nodes[1].factor_nodes) == 3
    assert len(graph.label_nodes[2].factor_nodes) == 3
    assert len(graph.label_nodes[3].factor_nodes) == 3
    assert len(graph.label_nodes[4].factor_nodes) == 3
    
    unary_nodes = graph.factor_nodes["unary"]
    assert len(unary_nodes) == 5
    for i, node in enumerate(unary_nodes):
        assert node.score_table_shape == [3]
        assert len(node.label_nodes) == 1
        assert node.label_nodes[0].node_id == i
        assert node in graph.label_nodes[i].factor_nodes
    assert graph.factor_label_indices["unary"] == [[0, 1, 2, 3, 4]]
    
    binary_nodes = graph.factor_nodes["binary"]
    assert len(binary_nodes) == 3
    assert [n.node_id for n in binary_nodes[0].label_nodes] == [1, 4]
    assert [n.node_id for n in binary_nodes[1].label_nodes] == [2, 3]
    assert [n.node_id for n in binary_nodes[2].label_nodes] == [3, 4]
    for factor_node in binary_nodes:
        assert factor_node.score_table_shape == [3, 3]
        for label_node in factor_node.label_nodes:
            assert factor_node in label_node.factor_nodes
    assert graph.factor_label_indices["binary"] == \
        [[1, 2, 3], [4, 3, 4]]

    tertiary_nodes = graph.factor_nodes["tertiary"]
    assert len(tertiary_nodes) == 1
    assert [n.node_id for n in tertiary_nodes[0].label_nodes] == [0, 1, 2]
    assert tertiary_nodes[0].score_table_shape == [3, 3, 3]
    for label_node in tertiary_nodes[0].label_nodes:
        assert tertiary_nodes[0] in label_node.factor_nodes
    assert graph.factor_label_indices["tertiary"] == [[0], [1], [2]]


def test_factor_graph_forward(five_node_graph):
    graph = five_node_graph
    input_states = torch.rand(5, 10)  # state_dim of the models is 10
    
    graph.forward(input_states)

    for node in graph.factor_nodes["unary"]:
        assert node.score_table.shape == (3, )
    for node in graph.factor_nodes["binary"]:
        assert node.score_table.shape == (3, 3)
    for node in graph.factor_nodes["tertiary"]:
        assert node.score_table.shape == (3, 3, 3)

    assert graph.current_score is None
    graph.random_init_labels()
    assert graph.current_score is not None


def test_factor_graph_compute_scores(five_node_graph_with_score):
    graph = five_node_graph_with_score
    graph.set_label_values([1, 0, 2, 1, 2])
    assert graph.current_score.item() == 31
    
    assert graph.compute_score_on_labels([0, 0, 0, 0, 0]).item() == 0
    assert graph.compute_score_on_labels([1, 0, 2, 1, 2]).item() == 31
    assert graph.compute_score_on_labels([2, 1, 0, 0, 1]).item() == 30
    assert graph.current_score.item() == 31  # Graph state is not changed


def test_init_with_oracle():
    graph = FactorGraph([10] * 100, dict(), dict())
    ground_truth = [random.randint(0, 9) for _ in range(100)]
    
    # Random init, mean 0.1
    graph.random_init_labels()
    assert sum([p == t for p, t in zip(graph.get_label_values(),
                                       ground_truth)]) < 30
    
    graph.random_init_labels(ground_truth, 1.0)
    assert sum([p == t for p, t in zip(graph.get_label_values(),
                                       ground_truth)]) == 100

    graph.random_init_labels(ground_truth, 0.5)
    assert sum([p == t for p, t in zip(graph.get_label_values(),
                                       ground_truth)]) > 30
