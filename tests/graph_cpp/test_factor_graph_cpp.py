"""Test the FactorGraphCpp class
"""

import pytest

import torch

from nsr.model.unary_factor import UnaryFactor
from nsr.model.high_order_factor import HighOrderFactor
from nsr.graph_cpp.factor_graph_cpp import FactorGraphCpp


@pytest.fixture()
def five_node_graph_cpp():
    """A dummy factor graph with 5 RV nodes.
           [*] ----                [*]
         /  |  ___ | ____________/  |
       /    | /    |                |
    (0)    (1)    (2)-[*]-(3)-[*]-(4)
     |      |      |       |       |
    [*]    [*]    [*]     [*]     [*]
    """
    state_dim = 10
    label_space_size = 3
    feature_ops = [lambda x: torch.sum(x, dim=0),
                   lambda x: torch.max(x, dim=0)[0]]
    return FactorGraphCpp(
        factor_dependencies={
            "unary": [(0,), (1,), (2,), (3,), (4,)],
            "binary": [(1, 4), (2, 3), (3, 4)],
            "tertiary": [(0, 1, 2)]},
        factor_models={
            "unary": UnaryFactor(state_dim, label_space_size, 15, 0.5),
            "binary": HighOrderFactor(state_dim, 2, feature_ops,
                                      label_space_size ** 2, 15, 0.5),
            "tertiary": HighOrderFactor(state_dim, 3, feature_ops,
                                        label_space_size ** 3, 15, 0.5)}
    )


def test_graph_cpp_factor_indices(five_node_graph_cpp):
    graph = five_node_graph_cpp
    
    assert len(list(graph.parameters())) > 0  # Sub-modules are registered.
    assert graph.factor_dependencies == {
            "unary": [(0,), (1,), (2,), (3,), (4,)],
            "binary": [(1, 4), (2, 3), (3, 4)],
            "tertiary": [(0, 1, 2)]}
    
    assert graph.factor_label_indices["unary"] == [[0, 1, 2, 3, 4]]
    assert graph.factor_label_indices["binary"] == \
        [[1, 2, 3], [4, 3, 4]]
    assert graph.factor_label_indices["tertiary"] == [[0], [1], [2]]


def test_graph_cpp_factor_forward(five_node_graph_cpp):
    graph = five_node_graph_cpp
    input_states = torch.rand(5, 10)  # state_dim of the models is 10
    factor_values = graph(input_states)
    
    assert len(factor_values) == 3
    assert factor_values["unary"].shape == (5, 3)
    assert factor_values["binary"].shape == (3, 9)
    assert factor_values["tertiary"].shape == (1, 27)
