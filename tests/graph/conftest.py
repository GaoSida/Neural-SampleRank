import pytest

import torch

from nsr.model.unary_factor import UnaryFactor
from nsr.model.high_order_factor import HighOrderFactor
from nsr.graph.factor_graph import LabelNode, FactorNode, FactorGraph


@pytest.fixture()
def unary_factor_node():
    factor_node = FactorNode([LabelNode(node_id=3, label_space_size=9)])
    # Let score values to be tied to index values
    factor_node.set_score_table(torch.tensor(range(9)))
    return factor_node


@pytest.fixture()
def binary_factor_node():
    factor_node = FactorNode([
        LabelNode(node_id=3, label_space_size=9),
        LabelNode(node_id=24, label_space_size=7)
    ])
    factor_node.set_score_table(torch.tensor(range(63)))
    return factor_node


@pytest.fixture()
def five_node_graph():
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
    return FactorGraph(
        label_space_sizes=[label_space_size] * 5,
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


@pytest.fixture()
def five_node_graph_with_score(five_node_graph):
    graph = five_node_graph
    for node in graph.factor_nodes["unary"]:
        node.score_table = torch.tensor(range(3))
    for node in graph.factor_nodes["binary"]:
        node.score_table = torch.tensor(range(9)).view(3, 3)
    for node in graph.factor_nodes["tertiary"]:
        node.score_table = torch.tensor(range(27)).view(3, 3, 3)
    return graph
