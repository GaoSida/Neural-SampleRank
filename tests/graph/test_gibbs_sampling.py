"""Test the standard Gibbs Sampling algorithms
"""
import torch

from nsr.graph.gibbs_sampling import compute_conditional_scores
from nsr.graph.gibbs_sampling import gibbs_sample_step


def test_unary_node_conditional_scores(unary_factor_node):
    factor_node = unary_factor_node

    conditional_score = compute_conditional_scores(factor_node, [3])
    assert conditional_score.tolist() == list(range(9))
    # The label being sampled is a superset
    conditional_score = compute_conditional_scores(factor_node, [1, 3, 5])
    assert conditional_score.shape == (1, 9, 1)
    assert conditional_score.flatten().tolist() == list(range(9))
    # Degenerate case:
    factor_node.label_nodes[0].current_value = 3
    assert factor_node.update_score().item() == 3
    conditional_score = compute_conditional_scores(factor_node, [5, 7])
    assert conditional_score.shape == (1, 1)
    assert conditional_score.item() == factor_node.current_score.item() == 3
    

def test_binary_node_conditional_scores(binary_factor_node):
    factor_node = binary_factor_node
    factor_node.label_nodes[0].current_value = 5
    factor_node.label_nodes[1].current_value = 2
    factor_node.update_score()

    conditional_score = compute_conditional_scores(factor_node, [3])
    assert conditional_score.shape == (9, )
    assert conditional_score.tolist() == [2, 9, 16, 23, 30, 37, 44, 51, 58]
    
    conditional_score = compute_conditional_scores(factor_node, [24])
    assert conditional_score.shape == (7, )
    assert conditional_score.tolist() == [35, 36, 37, 38, 39, 40, 41]
    
    conditional_score = compute_conditional_scores(factor_node, [3, 24])
    assert conditional_score.shape == (9, 7)
    assert (conditional_score == factor_node.score_table).all()
    
    conditional_score = compute_conditional_scores(factor_node, [24, 3])
    assert conditional_score.shape == (7, 9)
    assert (conditional_score == factor_node.score_table.t()).all()
    
    conditional_score = compute_conditional_scores(factor_node, [5, 3, 19, 24])
    assert conditional_score.shape == (1, 9, 1, 7)
    assert (conditional_score.view(9, 7) == factor_node.score_table).all()
    # Degenerate case
    conditional_score = compute_conditional_scores(factor_node, [5, 7, 19])
    assert conditional_score.shape == (1, 1, 1)
    assert conditional_score.item() == 37


def test_gibbs_sample_step(five_node_graph_with_score):
    """Test the conditional scores and (forced) sample results.
    """
    graph = five_node_graph_with_score
    
    # Check the conditional scores
    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [0])
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([2, 12, 22])).all()

    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [0], temp=0.1)
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([20, 120, 220])).all()

    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [1])
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([13, 20, 27])).all()
    
    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [3])
    assert conditional_scores.shape == (3, )
    assert (conditional_scores == torch.tensor([8, 13, 18])).all()
    
    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [1, 2])
    assert conditional_scores.shape == (3, 3)
    assert (conditional_scores == torch.tensor([
        [12, 17, 22], [19, 24, 29], [26, 31, 36]
    ])).all()
    
    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [2, 1])
    assert conditional_scores.shape == (3, 3)
    assert (conditional_scores == torch.tensor([
        [12, 19, 26], [17, 24, 31], [22, 29, 36]
    ])).all()
    
    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [3, 4])
    assert conditional_scores.shape == (3, 3)
    assert (conditional_scores == torch.tensor([
        [6, 9, 12], [11, 14, 17], [16, 19, 22]
    ])).all()
    
    graph.set_label_values([1, 0, 2, 1, 2])
    conditional_scores = gibbs_sample_step(graph, [0, 1, 2])
    assert conditional_scores.shape == (3, 3, 3)
    assert (conditional_scores == torch.tensor([
        [[3, 8, 13], [10, 15, 20], [17, 22, 27]],
        [[13, 18, 23], [20, 25, 30], [27, 32, 37]],
        [[23, 28, 33], [30, 35, 40], [37, 42, 47]]
    ])).all()
    
    # Check the forced samples (and thus label index parsing)
    graph.set_label_values([1, 0, 2, 1, 2])
    table = torch.zeros(3, 3, 3, dtype=torch.long)
    table[2, 1, 0] = 200
    graph.factor_nodes["tertiary"][0].score_table = table
    gibbs_sample_step(graph, [0, 2, 1])
    assert graph.get_label_values() == [2, 1, 0, 1, 2]
    
    # Note that current state and score are dirty
    table = torch.zeros(3, 3, dtype=torch.long)
    table[2, 1] = 200
    graph.factor_nodes["binary"][2].score_table = table
    gibbs_sample_step(graph, [3, 4])
    assert graph.get_label_values() == [2, 1, 0, 2, 1]

    graph.factor_nodes["binary"][2].score_table[2, 2] = 500
    gibbs_sample_step(graph, [4])
    assert graph.get_label_values() == [2, 1, 0, 2, 2]
    
    graph.factor_nodes["unary"][2].score_table = torch.tensor([0, 300, 0])
    gibbs_sample_step(graph, [2])
    assert graph.get_label_values() == [2, 1, 1, 2, 2]
    
    gibbs_sample_step(graph, [3, 2])
    assert graph.get_label_values() == [2, 1, 1, 2, 2]
